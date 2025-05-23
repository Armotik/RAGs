# preprocessing/qa_handler.py
from typing import List, Dict, Any
import torch
from tqdm import tqdm
from bert_score import score as bert_scorer
import re
import os


# --- Helper Function: Création du Prompt pour Q/A (MODIFIÉ pour plus de directivité et un exemple) ---
def _create_qa_generation_prompt(text_segment: str, title: str, entities: list, lang:str) -> str:
    if entities and isinstance(entities[0], tuple):
        entities_str = ", ".join([str(ent[0]) for ent in entities if ent and ent[0]])
    elif entities and isinstance(entities[0], str):
        entities_str = ", ".join(entities)
    else:
        entities_str = "aucune entité spécifique notable"

    max_segment_display_len = 1500
    text_segment_for_prompt_display = text_segment
    if len(text_segment) > max_segment_display_len:
        text_segment_for_prompt_display = text_segment[
                                          :max_segment_display_len] + " ... (segment tronqué pour affichage prompt)"

    prompt = f"""Tu es un assistant expert en analyse de texte. Ta tâche est de générer une question, sa réponse, et un score de confiance à partir de l'extrait de texte fourni.
L'extrait provient de l'article intitulé "{title}" et mentionne potentiellement les entités : [{entities_str}].

Extrait de texte à analyser :
---
{text_segment_for_prompt_display}
---

Instructions STRICTES pour la génération :
1.  **Question :** Doit commencer EXACTEMENT par "Question: ". Génère UNE question concise et pertinente à laquelle l'extrait répond. Si l'extrait ne permet pas de formuler une question claire et une réponse factuelle, écris "Question: N/A". NE PAS NUMÉROTER cette ligne.
2.  **Réponse :** Doit commencer EXACTEMENT par "Réponse: ". Fournis la réponse directe et concise, basée UNIQUEMENT sur l'extrait. Si la Question est N/A, écris "Réponse: N/A". NE PAS NUMÉROTER cette ligne.
3.  **Confiance\_LLM :** Doit commencer EXACTEMENT par "Confiance\_LLM: ". Fournis un score flottant entre 0.0 et 1.0. Si la Question est N/A, écris "Confiance\_LLM: 0.0". NE PAS NUMÉROTER cette ligne. NE RIEN AJOUTER sur cette ligne après le score numérique.
4.  **Langue :** Il faut générer la question et la réponse dans cette langue : {lang}

Voici un EXEMPLE de format de sortie PARFAITEMENT RESPECTÉ :
Question: Quelle est la capitale de la France ?
Réponse: Paris.
Confiance_LLM: 0.9

Ta production doit suivre IMPÉRATIVEMENT ce format, chaque item sur une nouvelle ligne distincte :
"""
    # Laisser le LLM compléter après "Ta production :"
    return prompt


# --- Helper Function: Parsing de la Sortie LLM (MODIFIÉE pour plus de robustesse) ---
def _parse_llm_qa_output(llm_output: str) -> dict:
    question = None
    answer = None
    llm_confidence = 0.0

    # Nettoyer les lignes vides potentielles au début et à la fin, et les espaces en début/fin de chaque ligne
    cleaned_lines = [line.strip() for line in llm_output.strip().splitlines() if line.strip()]
    cleaned_output_for_regex = "\n".join(cleaned_lines)

    # DEBUG:
    # print(f"--- _parse_llm_qa_output: Tentative de parsing de (nettoyé) ---\n{cleaned_output_for_regex}\n---")

    # Regex pour capturer le contenu après les tags.
    # On s'attend à ce que chaque tag soit au début de sa ligne.
    # (.*) capturera le reste de la ligne.
    q_match = re.search(r"^Question:\s*(.*)", cleaned_output_for_regex, re.MULTILINE | re.IGNORECASE)
    a_match = re.search(r"^Réponse:\s*(.*)", cleaned_output_for_regex, re.MULTILINE | re.IGNORECASE)
    # Pour la confiance, capturer le nombre, tolérer des points de suspension ou autre chose après, mais on ne prend que le nombre.
    c_match = re.search(r"^Confiance_LLM:\s*([0-1](?:\.[0-9]+)?)(?:.*)", cleaned_output_for_regex,
                        re.MULTILINE | re.IGNORECASE)  # Modifié ici

    if q_match:
        question_content = q_match.group(1).strip()
        if question_content.upper() == "N/A":
            question = None
        else:
            question = question_content
    else:
        if "N/A" not in cleaned_output_for_regex.upper():  # Ne pas loguer si c'est un N/A global
            print(
                f"[AVERTISSEMENT _parse_llm_qa_output] Tag 'Question:' non trouvé. Sortie (début): '{cleaned_output_for_regex[:200].replace(os.linesep, ' ')}...'")

    if a_match:
        answer_content = a_match.group(1).strip()
        if answer_content.upper() == "N/A":
            answer = None
        else:
            answer = answer_content
    else:
        if question is not None and "N/A" not in cleaned_output_for_regex.upper():
            print(
                f"[AVERTISSEMENT _parse_llm_qa_output] Tag 'Réponse:' non trouvé (Question était: '{str(question)[:50]}...'). Sortie (début): '{cleaned_output_for_regex[:200].replace(os.linesep, ' ')}...'")

    if c_match:
        try:
            llm_confidence = float(c_match.group(1).strip())
        except ValueError:
            print(
                f"[AVERTISSEMENT _parse_llm_qa_output] Impossible de convertir score Confiance_LLM: '{c_match.group(1).strip()}'. Utilisation de 0.0.")
            llm_confidence = 0.0
    else:
        if question is None and answer is None:  # Si Q et A sont N/A (donc None ici)
            llm_confidence = 0.0
        elif "N/A" not in cleaned_output_for_regex.upper():  # Si ce n'est pas un N/A global
            print(
                f"[AVERTISSEMENT _parse_llm_qa_output] Tag 'Confiance_LLM:' non trouvé ou format score incorrect. Sortie (début): '{cleaned_output_for_regex[:200].replace(os.linesep, ' ')}...'. Confiance mise à 0.0.")
            llm_confidence = 0.0

    if question is None and answer is None and llm_confidence == 0.0:
        raw_output_upper = cleaned_output_for_regex.upper()
        is_explicit_na_response = ("QUESTION: N/A" in raw_output_upper and
                                   "RÉPONSE: N/A" in raw_output_upper and
                                   ("CONFIANCE_LLM: 0.0" in raw_output_upper or "CONFIANCE_LLM: 0" in raw_output_upper))

        if not is_explicit_na_response and raw_output_upper.strip() != "N/A":
            print(
                f"[AVERTISSEMENT _parse_llm_qa_output] Aucun champ Q, A valide n'a été parsé et ce n'est pas un N/A explicite. Format LLM probablement incorrect. Sortie NETTOYÉE (début): '{cleaned_output_for_regex[:300].replace(os.linesep, ' ')}...'")

    return {
        "generated_question": question,
        "generated_answer": answer,
        "llm_confidence": llm_confidence
    }


# --- Fonction Principale de Traitement en Batchs ---
def process_segments_in_batches(
        segments_data: List[Dict],
        llm_qab_model: Any,
        llm_qab_tokenizer: Any,
        bert_score_model_type: str = "bert-base-multilingual-cased",
        batch_size_llm: int = 8,  # RÉDUIT POUR DÉBOGAGE FIN, AUGMENTE SUR H100 (ex: 64, 128)
        batch_size_bertscore: int = 32,  # Idem
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens_qa: int = 300,
        temperature_qa: float = 0.3  # Gardé bas pour plus de respect du format
) -> List[Dict]:
    # ... (Début de la fonction inchangé : initialisation de all_processed_segments, boucle tqdm) ...
    all_processed_segments = []
    num_segments = len(segments_data)

    if llm_qab_tokenizer.pad_token is None:
        llm_qab_tokenizer.pad_token = llm_qab_tokenizer.eos_token
        print(
            f"[INFO process_segments_in_batches] pad_token_id for llm_qab_tokenizer set to eos_token_id: {llm_qab_tokenizer.eos_token_id}")
    if llm_qab_tokenizer.padding_side != 'left':
        print(
            f"[AVERTISSEMENT process_segments_in_batches] llm_qab_tokenizer.padding_side n'est pas 'left' ({llm_qab_tokenizer.padding_side}). Pour la génération en batch, 'left' est souvent recommandé et a été défini dans le script principal.")

    for i in tqdm(range(0, num_segments, batch_size_llm), desc="Traitement des lots de segments pour Q/A"):
        current_batch_input_data = segments_data[i:i + batch_size_llm]
        if not current_batch_input_data:
            continue

        prompts_batch = []
        valid_segments_for_llm_call = []

        for seg_data in current_batch_input_data:
            text_s = seg_data.get('text_segment', "")
            if text_s and text_s.strip() and len(text_s.split()) > 7:
                prompts_batch.append(
                    _create_qa_generation_prompt(text_s, seg_data.get('title', 'N/A'), seg_data.get('entities', []),
                                                 seg_data.get('lang', 'fr'))
                )
                valid_segments_for_llm_call.append(seg_data)
            else:
                all_processed_segments.append({
                    **seg_data, "generated_question": None, "generated_answer": None,
                    "llm_confidence": 0.0, "bert_score_f1": 0.0,
                    "error_qab_generation": "Segment initial vide ou trop court (< ~7 mots)"
                })

        if not prompts_batch:
            continue

        if i == 0:
            print(
                f"\n--- Prompt Q/A (segment 0 du batch 0) ---\n{prompts_batch[0]}\n------------------------------------------\n")

        inputs = llm_qab_tokenizer(
            prompts_batch, return_tensors="pt", padding=True, truncation=True,
            max_length=1800  # Ajusté, dépend de la longueur max de tes segments + prompt
        ).to(device)

        batch_responses_text_list = [""] * len(prompts_batch)
        try:
            generated_ids_batch = llm_qab_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens_qa,
                temperature=temperature_qa,
                do_sample=(temperature_qa > 0.0),  # do_sample=True si temperature > 0
                pad_token_id=llm_qab_tokenizer.pad_token_id,
                eos_token_id=llm_qab_tokenizer.eos_token_id,
                # repetition_penalty=1.1 # MODIFICATION : Peut aider à éviter les répétitions et forcer le format
            )

            for k_resp in range(generated_ids_batch.shape[0]):
                start_decode_index = inputs['input_ids'].shape[1]
                # MODIFICATION : Ajout de clean_up_tokenization_spaces=True
                decoded_text = llm_qab_tokenizer.decode(
                    generated_ids_batch[k_resp, start_decode_index:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()
                batch_responses_text_list[k_resp] = decoded_text

                # MODIFICATION : Affiche la sortie brute pour TOUS les items du PREMIER BATCH pour un meilleur débogage
                if i == 0:
                    print(
                        f"\n--- Sortie Brute LLM (batch 0, segment {k_resp} / ID original: {valid_segments_for_llm_call[k_resp].get('original_doc_id', 'N/A')}) ---\n{decoded_text}\n--------------------------------------------------\n")

        except Exception as e:
            print(f"[ERREUR] Échec de la génération LLM pour batch {i // batch_size_llm} : {e}")
            # ... (gestion d'erreur comme avant) ...
            for seg_data_error in valid_segments_for_llm_call:
                all_processed_segments.append({
                    **seg_data_error, "generated_question": None, "generated_answer": None,
                    "llm_confidence": None,
                    "bert_score_f1": 0.0, "error_qab_generation": str(e)
                })
            continue

        parsed_qas_batch = [_parse_llm_qa_output(output_txt) for output_txt in batch_responses_text_list]

        # ... (Partie BERTScore et assemblage des résultats - reste identique à ma réponse précédente) ...
        original_texts_for_bertscore = [seg['text_segment'] for seg in valid_segments_for_llm_call]
        generated_answers_for_bertscore = [pqa['generated_answer'] if pqa and pqa.get('generated_answer') else "" for
                                           pqa in parsed_qas_batch]
        langs_for_bertscore = [seg.get('lang', 'fr') for seg in valid_segments_for_llm_call]

        bert_f1_scores_for_batch = [0.0] * len(valid_segments_for_llm_call)

        indices_with_valid_answers = [k for k, ans in enumerate(generated_answers_for_bertscore) if
                                      ans.strip() and original_texts_for_bertscore[k].strip()]

        if indices_with_valid_answers:
            answers_to_score = [generated_answers_for_bertscore[k] for k in indices_with_valid_answers]
            originals_to_score = [original_texts_for_bertscore[k] for k in indices_with_valid_answers]

            unique_langs_in_batch = sorted(list(set(
                lang for idx_lang, lang in enumerate(langs_for_bertscore) if idx_lang in indices_with_valid_answers)))

            for lang_bs in unique_langs_in_batch:
                lang_specific_indices_in_batch = [k for k in indices_with_valid_answers if
                                                  langs_for_bertscore[k] == lang_bs]
                if not lang_specific_indices_in_batch: continue

                lang_answers = [generated_answers_for_bertscore[k] for k in lang_specific_indices_in_batch]
                lang_originals = [original_texts_for_bertscore[k] for k in lang_specific_indices_in_batch]

                try:
                    if lang_answers and lang_originals:
                        _, _, f1_temp_lang = bert_scorer(
                            lang_answers, lang_originals, lang=lang_bs,
                            model_type=bert_score_model_type, device=device,
                            batch_size=batch_size_bertscore, verbose=False
                        )
                        f1_iter_lang = iter(f1_temp_lang)
                        for real_idx_in_valid_segments in lang_specific_indices_in_batch:
                            bert_f1_scores_for_batch[real_idx_in_valid_segments] = round(next(f1_iter_lang).item(), 4)
                except Exception as e:
                    print(f"[ERREUR] Calcul BERTScore pour batch (lang: {lang_bs}) échoué : {e}")

        for idx_in_batch, seg_data_orig in enumerate(valid_segments_for_llm_call):
            parsed_qa_current = parsed_qas_batch[idx_in_batch] if idx_in_batch < len(parsed_qas_batch) else {
                "generated_question": None, "generated_answer": None, "llm_confidence": 0.0}
            all_processed_segments.append({
                **seg_data_orig,
                "generated_question": parsed_qa_current["generated_question"],
                "generated_answer": parsed_qa_current["generated_answer"],
                "llm_confidence": parsed_qa_current["llm_confidence"],
                "bert_score_f1": bert_f1_scores_for_batch[idx_in_batch]
            })

    return all_processed_segments