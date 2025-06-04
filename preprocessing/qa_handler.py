# preprocessing/qa_handler.py
from typing import List, Dict, Any
import torch
from tqdm import tqdm
from bert_score import score as bert_scorer
import re
import os


def _create_qa_generation_prompt(text_segment: str, title: str, entities: list, lang: str) -> str:
    """
    Create a prompt for the LLM to generate a question and answer based on the provided text segment, title, entities, and language.
    :param text_segment: The text segment to analyze.
    :param title: The title of the article or document from which the text segment is extracted.
    :param entities: A list of entities mentioned in the text segment, which can be tuples or strings.
    :param lang: The language in which the question and answer should be generated (e.g., 'fr' for French).
    :return: A formatted prompt string for the LLM.
    """
    if entities and isinstance(entities[0], tuple):
        entities_str = ", ".join([str(ent[0]) for ent in entities if ent and ent[0]])
    elif entities and isinstance(entities[0], str):
        entities_str = ", ".join(entities)
    else:
        entities_str = "aucune entité spécifique notable"

    max_segment_display_len = 1800
    text_segment_for_prompt_display = text_segment
    if len(text_segment) > max_segment_display_len:
        text_segment_for_prompt_display = text_segment[
                                          :max_segment_display_len] + " ... (segment tronqué pour affichage prompt)"

    prompt = f"""Tu es un assistant expert en analyse de texte, programmé pour formuler des questions pertinentes et naturelles à partir d'extraits de documents.
L'article original concerne "{title}". L'extrait actuel que tu dois analyser mentionne potentiellement les entités suivantes : [{entities_str}].

Extrait de texte à analyser :
---
{text_segment_for_prompt_display}
---

INSTRUCTIONS DÉTAILLÉES ET IMPÉRATIVES POUR TA PRODUCTION :

1.  **Génération de la Question :**
    * Ta question DOIT commencer EXACTEMENT par la chaîne "Question: " (sans numéro devant).
    * Formule UNE question claire, concise, et précise qui porte sur une information saillante, spécifique et non triviale de l'extrait.
    * **Important :** La question doit être formulée de manière naturelle, comme si un utilisateur la posait pour obtenir une information, SANS faire référence explicitement à l'extrait lui-même. La question doit pouvoir être comprise de manière autonome.
    * Si possible, la question devrait impliquer le sujet principal de l'article ("{title}") ou les entités fournies ([{entities_str}]), en cherchant à clarifier leur rôle, leurs actions, ou leurs liens.
    * Si l'extrait est trop court, vide, ou ne contient aucune information substantielle permettant de formuler une question de qualité respectant ces critères, la ligne entière doit être "Question: N/A".
    * NE PAS numéroter cette ligne "Question:".

2.  **Génération de la Réponse :**
    * Ta réponse DOIT commencer EXACTEMENT par la chaîne "Réponse: " (sans numéro devant), sur une nouvelle ligne distincte immédiatement après la question.
    * Fournis une réponse directe, factuelle et concise à la question que tu as formulée.
    * La réponse DOIT être extraite ou directement et sans ambiguïté inférable EXCLUSIVEMENT à partir des informations présentes dans l'"Extrait de texte à analyser" fourni ci-dessus. N'utilise AUCUNE connaissance extérieure.
    * Si la question est "N/A", la ligne entière pour la réponse DOIT être "Réponse: N/A".
    * NE PAS numéroter cette ligne "Réponse:".

3.  **Score de Confiance :**
    * Ton score de confiance DOIT commencer EXACTEMENT par la chaîne "Confiance_LLM: " (sans numéro devant), sur une nouvelle ligne distincte immédiatement après la réponse.
    * Attribue un score numérique flottant unique (par exemple 0.75, 0.9, 1.0) entre 0.0 et 1.0. Ce score doit refléter ta confiance absolue dans la clarté et la non-ambiguïté avec lesquelles l'extrait supporte la paire question/réponse que tu as générée.
    * Si la question est "N/A", le score de confiance DOIT être "0.0".
    * NE PAS numéroter cette ligne "Confiance_LLM:". NE RIEN ajouter après le score numérique sur cette ligne.

4.  **Langue :** La question et la réponse DOIVENT être générées dans la langue suivante : {lang}.

RAPPEL DU FORMAT DE SORTIE STRICTEMENT ATTENDU (TROIS LIGNES DISTINCTES UNIQUEMENT) :
Question: [Contenu de ta question naturelle et spécifique ou N/A]
Réponse: [Contenu de ta réponse concise et factuelle ou N/A]
Confiance_LLM: [Score numérique entre 0.0 et 1.0]

Commence ta production directement par "Question:".
"""
    return prompt


def _parse_llm_qa_output(llm_output: str) -> dict:
    """
    Parse the output from the LLM to extract the generated question, answer, and confidence score.
    :param llm_output: The raw output string from the LLM, which should contain the question, answer, and confidence score.
    :return: A dictionary containing the parsed question, answer, and confidence score.
    """
    question = None
    answer = None
    llm_confidence = 0.0

    cleaned_lines = [line.strip() for line in llm_output.strip().splitlines() if line.strip()]
    cleaned_output_for_regex = "\n".join(cleaned_lines)

    q_match = re.search(r"^(?:[0-9]+\s*[\.\-]?\s*)?Question:\s*(.*)", cleaned_output_for_regex,
                        re.MULTILINE | re.IGNORECASE)
    a_match = re.search(r"^(?:[0-9]+\s*[\.\-]?\s*)?Réponse:\s*(.*)", cleaned_output_for_regex,
                        re.MULTILINE | re.IGNORECASE)
    c_match = re.search(r"^(?:[0-9]+\s*[\.\-]?\s*)?Confian(?:ce|dence)_LLM:\s*([0-1](?:\.[0-9]+)?)(?:.*)",
                        cleaned_output_for_regex, re.MULTILINE | re.IGNORECASE)

    if q_match:
        question_content = q_match.group(1).strip()
        if question_content.upper() == "N/A":
            question = None
        else:
            question = question_content

    if a_match:
        answer_content = a_match.group(1).strip()
        if answer_content.upper() == "N/A":
            answer = None
        else:
            answer = answer_content

    if c_match:
        try:
            llm_confidence = float(c_match.group(1))
        except ValueError:
            llm_confidence = 0.0
    else:
        if question is None and answer is None:
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


def process_segments_in_batches(
        segments_data: List[Dict],
        llm_qab_model: Any,
        llm_qab_tokenizer: Any,
        bert_score_model_type: str = "bert-base-multilingual-cased",
        batch_size_llm: int = 8,
        batch_size_bertscore: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens_qa: int = 250,
        temperature_qa: float = 0.3
) -> List[Dict]:
    """
Process segments of text data in batches to generate questions and answers using a language model (LLM) and evaluate them with BERTScore.
    :param segments_data: List of dictionaries containing segment data, each with keys like 'text_segment', 'title', 'entities', and 'lang'.
    :param llm_qab_model: The language model to use for question and answer generation.
    :param llm_qab_tokenizer: The tokenizer corresponding to the language model for question and answer generation.
    :param bert_score_model_type: A string indicating the BERT model type to use for scoring (default is "bert-base-multilingual-cased").
    :param batch_size_llm: The batch size for processing segments with the LLM.
    :param batch_size_bertscore: The batch size for calculating BERTScore.
    :param device: The device to run the model on, either 'cuda' for GPU or 'cpu'.
    :param max_new_tokens_qa: The maximum number of new tokens to generate for each question and answer pair.
    :param temperature_qa: The temperature parameter for controlling the randomness of the LLM output (default is 0.3).
    :return: A list of dictionaries containing the original segment data along with the generated question, answer, LLM confidence score, and BERTScore F1 score.
    """

    all_processed_segments = []
    num_segments = len(segments_data)

    if llm_qab_tokenizer.pad_token is None:
        llm_qab_tokenizer.pad_token = llm_qab_tokenizer.eos_token
        print(
            f"[INFO process_segments_in_batches] pad_token_id for llm_qab_tokenizer set to eos_token_id: {llm_qab_tokenizer.eos_token_id}")

    for i in tqdm(range(0, num_segments, batch_size_llm), desc="Traitement des lots de segments pour Q/A"):
        current_batch_input_data = segments_data[i:i + batch_size_llm]
        if not current_batch_input_data:
            continue

        prompts_batch = []
        valid_segments_for_llm_call = []

        for seg_data in current_batch_input_data:
            text_s = seg_data.get('text_segment', "")
            if text_s and text_s.strip():
                prompts_batch.append(
                    _create_qa_generation_prompt(text_s,
                                                 seg_data.get('title', 'Titre inconnu'),
                                                 seg_data.get('entities', []),
                                                 seg_data.get('lang', 'fr'))
                )
                valid_segments_for_llm_call.append(seg_data)
            else:
                all_processed_segments.append({
                    **seg_data, "generated_question": None, "generated_answer": None,
                    "llm_confidence": 0.0, "bert_score_f1": 0.0,
                    "error_qab_generation": "Texte segment vide ou invalide"
                })

        if not prompts_batch:
            continue

        if i == 0 and prompts_batch:
            print(
                f"\n--- Prompt Q/A (segment 0 du batch 0) ---\n{prompts_batch[0]}\n------------------------------------------\n")

        inputs = llm_qab_tokenizer(
            prompts_batch, return_tensors="pt", padding=True, truncation=True,
            max_length=1800
        ).to(device)

        batch_responses_text_list = [""] * len(prompts_batch)
        try:
            generated_ids_batch = llm_qab_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens_qa,
                temperature=temperature_qa,
                do_sample=(temperature_qa > 0.001),
                pad_token_id=llm_qab_tokenizer.pad_token_id,
                eos_token_id=llm_qab_tokenizer.eos_token_id
            )

            for k_resp in range(generated_ids_batch.shape[0]):
                start_decode_index = inputs['input_ids'].shape[1]
                decoded_text = llm_qab_tokenizer.decode(
                    generated_ids_batch[k_resp, start_decode_index:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()
                batch_responses_text_list[k_resp] = decoded_text

                if i == 0:
                    print(
                        f"\n--- Sortie Brute LLM (batch 0, segment {k_resp} / ID original: {valid_segments_for_llm_call[k_resp].get('original_doc_id', 'N/A')}) ---\n{decoded_text}\n--------------------------------------------------\n")

        except Exception as e:
            print(f"[ERREUR] Échec de la génération LLM pour batch {i // batch_size_llm} : {e}")
            for seg_data_error in valid_segments_for_llm_call:
                all_processed_segments.append({
                    **seg_data_error, "generated_question": None, "generated_answer": None,
                    "llm_confidence": None,
                    "bert_score_f1": 0.0, "error_qab_generation": str(e)
                })
            continue

        parsed_qas_batch = [_parse_llm_qa_output(output_txt) for output_txt in batch_responses_text_list]

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
            langs_of_answers_to_score = [langs_for_bertscore[k] for k in indices_with_valid_answers]

            unique_langs_in_score_batch = sorted(list(set(langs_of_answers_to_score)))

            for lang_bs in unique_langs_in_score_batch:
                current_lang_sub_indices = [sub_idx for sub_idx, original_batch_idx in
                                            enumerate(indices_with_valid_answers) if
                                            langs_for_bertscore[original_batch_idx] == lang_bs]
                if not current_lang_sub_indices: continue

                lang_answers = [answers_to_score[sub_idx] for sub_idx in current_lang_sub_indices]
                lang_originals = [originals_to_score[sub_idx] for sub_idx in current_lang_sub_indices]

                try:
                    if lang_answers and lang_originals:
                        _, _, f1_temp_lang = bert_scorer(
                            lang_answers, lang_originals, lang=lang_bs,
                            model_type=bert_score_model_type, device=device,
                            batch_size=batch_size_bertscore, verbose=False
                        )
                        for list_idx_in_lang_batch, original_batch_idx_in_valid_segments in enumerate(
                                current_lang_sub_indices):
                            actual_index_in_bert_f1_scores = indices_with_valid_answers[
                                original_batch_idx_in_valid_segments]
                            bert_f1_scores_for_batch[actual_index_in_bert_f1_scores] = round(
                                f1_temp_lang[list_idx_in_lang_batch].item(), 4)
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