import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
import os
from typing import List, Dict, Any


def _create_qa_generation_prompt(text_segment: str, title: str, entities: list, lang: str) -> str:
    """
    Crée le prompt pour guider le LLM dans la génération de la Q/A.
    """
    if entities and isinstance(entities[0], tuple):
        entities_str = ", ".join([str(ent[0]) for ent in entities if ent and ent[0]])
    elif entities and isinstance(entities[0], str):
        entities_str = ", ".join(entities)
    else:
        entities_str = "aucune entité spécifique notable"

    max_segment_display_len = 1500
    text_segment_for_prompt_display = text_segment
    if len(text_segment) > max_segment_display_len:
        text_segment_for_prompt_display = text_segment[:max_segment_display_len] + " ... (segment tronqué)"

    prompt = f"""[INST] Tu es un assistant expert en analyse de texte. Ta tâche est de générer une question et sa réponse à partir de l'extrait de texte fourni.
L'extrait provient de l'article intitulé "{title}" et mentionne potentiellement les entités : [{entities_str}].

Extrait de texte à analyser :
---
{text_segment_for_prompt_display}
---

Instructions STRICTES pour la génération :
1.  **Question :** Doit commencer EXACTEMENT par "Question: ". Génère UNE question concise et pertinente à laquelle l'extrait répond. Si l'extrait ne permet pas de formuler une question claire, écris "Question: N/A".
2.  **Réponse :** Doit commencer EXACTEMENT par "Réponse: ". Fournis la réponse directe et concise, basée UNIQUEMENT sur l'extrait. Si la Question est N/A, écris "Réponse: N/A".

Ta production doit suivre IMPÉRATIVEMENT ce format, chaque item sur une nouvelle ligne distincte :
[/INST]
"""
    return prompt


def _parse_llm_qa_output(llm_output: str) -> dict:
    """
    Parse la sortie brute du LLM pour extraire la question et la réponse.
    """
    question, answer = None, None
    cleaned_lines = [line.strip() for line in llm_output.strip().splitlines() if line.strip()]
    cleaned_output_for_regex = "\n".join(cleaned_lines)

    q_match = re.search(r"^Question:\s*(.*)", cleaned_output_for_regex, re.MULTILINE | re.IGNORECASE)
    a_match = re.search(r"^Réponse:\s*(.*)", cleaned_output_for_regex, re.MULTILINE | re.IGNORECASE)

    if q_match:
        question_content = q_match.group(1).strip()
        if question_content.upper() != "N/A":
            question = question_content

    if a_match:
        answer_content = a_match.group(1).strip()
        if answer_content.upper() != "N/A":
            answer = answer_content

    # Si la question est nulle mais pas la réponse (cas étrange), on invalide la réponse
    if question is None:
        answer = None

    return {"generated_question": question, "generated_answer": answer}


def process_segments_for_qa(
        segments_data: List[Dict],
        model_id: str,
        device: str,
) -> List[Dict]:
    """
    Fonction principale du worker : charge un modèle et traite un lot de segments.
    """
    global tokenizer, model, batch_responses
    print(f"[Worker on {device}] Chargement du modèle {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        )
        print(f"[Worker on {device}] Modèle chargé et prêt.")
    except Exception as e:
        print(f"[Worker on {device}] ERREUR FATALE au chargement du modèle: {e}")
        return [{**seg, 'error_qab_generation': str(e)} for seg in segments_data]

    all_processed_segments = []

    prompts_batch = [
        _create_qa_generation_prompt(
            seg.get('text_segment', ''),
            seg.get('title', 'N/A'),
            seg.get('entities', []),
            seg.get('lang', 'fr')
        )
        for seg in segments_data
    ]

    try:
        inputs = tokenizer(prompts_batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(
            device)
        generated_ids_batch = model.generate(**inputs, max_new_tokens=256, temperature=0.2, do_sample=True)
        batch_responses = tokenizer.batch_decode(generated_ids_batch[:, inputs['input_ids'].shape[1]:],
                                                 skip_special_tokens=True)
    except Exception as e:
        print(f"[Worker on {device}] ERREUR lors de la génération LLM: {e}")
        return [{**seg, 'error_qab_generation': str(e)} for seg in segments_data]

    parsed_qas_batch = [_parse_llm_qa_output(output) for output in batch_responses]

    for i, seg_data in enumerate(segments_data):
        all_processed_segments.append({
            **seg_data,
            **parsed_qas_batch[i],
            "error_qab_generation": None
        })

    del model, tokenizer
    torch.cuda.empty_cache()
    return all_processed_segments