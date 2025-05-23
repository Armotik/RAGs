# preprocessing/pretraitment.py
import ast
import os
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
import torch
import datetime

from .qa_handler import process_segments_in_batches
from .chunking import chunking
from .content_extraction import load_lang
from .text_cleaning import clean_text
from .date_extraction import extract_date
from .NER import enrich_df_with_ner_pipe


def preprocess_data(
        languages: list[str],
        max_docs_per_lang: int,
        nb_chunk_for_parallel_ner: int,  # Nom clarifié
        llm_model_for_qa,
        llm_tokenizer_for_qa,
        new_docs=False
) -> pd.DataFrame:
    """
    Main function to preprocess documents for Q/A generation.
    :param languages: The list of languages to process.
    :param max_docs_per_lang: Maximum number of documents to process per language.
    :param nb_chunk_for_parallel_ner: Number of chunks for parallel NER processing.
    :param llm_model_for_qa: The language model for Q/A generation.
    :param llm_tokenizer_for_qa: The tokenizer for the language model.
    :param new_docs: If True, forces reprocessing of documents.
    :return: A DataFrame containing the processed documents with Q/A pairs.
    """
    t_time = datetime.datetime.now()
    save_dir = 'data'
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_file_path = os.path.join(save_dir, "df_contextual_chunks_qab_ready.parquet")
    checkpoint_flag_path = os.path.join(save_dir, "checkpoint__df_contextual_chunks_qab_ready.json")

    if os.path.exists(checkpoint_flag_path) and not new_docs:
        print(f"[INFO] Chargement des chunks contextuels Q/A depuis le checkpoint : {checkpoint_file_path}")
        try:
            df_contextual_chunks = pd.read_parquet(checkpoint_file_path)
            print(
                f"[INFO - {datetime.datetime.now()}] Documents (avec Q/A) chargés depuis checkpoint en {datetime.datetime.now() - t_time}")
            if not df_contextual_chunks.empty:
                return df_contextual_chunks
            else:
                print("[AVERTISSEMENT] Checkpoint Parquet trouvé mais vide. Recalcul.")
        except Exception as e:
            print(f"[AVERTISSEMENT] Échec du chargement du checkpoint Parquet {checkpoint_file_path}: {e}. Recalcul.")

    print(
        f"[INFO - {datetime.datetime.now()}] Début du prétraitement complet (pas de checkpoint valide ou new_docs=True).")


    print(f"[INFO - {datetime.datetime.now()}] Loading documents...")
    time_load = datetime.datetime.now()
    results_load = Parallel(n_jobs=len(languages))(
        delayed(load_lang)(lang, max_docs_per_lang) for lang in languages
    )
    all_docs = [doc for lang_docs in results_load for doc in lang_docs]
    print(f"[INFO - {datetime.datetime.now()}] Documents loaded in {datetime.datetime.now() - time_load}")

    df = pd.DataFrame(all_docs)
    if df.empty:
        print("[AVERTISSEMENT] Aucun document chargé. Arrêt du prétraitement.")
        return df

    df["title"] = df["title"].apply(clean_text)
    df["text"] = df["text"].apply(clean_text)

    df_initial_chunks = chunking(df, 0)
    if df_initial_chunks.empty:
        print("[AVERTISSEMENT] Aucun chunk initial après 'chunking'. Arrêt du prétraitement.")
        return df_initial_chunks

    print(f"[INFO - {datetime.datetime.now()}] Enriching metadata (dates) ...")
    time_date = datetime.datetime.now()
    df_initial_chunks = extract_date(df_initial_chunks)
    print(f"[INFO - {datetime.datetime.now()}] Metadata (dates) enriched in {datetime.datetime.now() - time_date}")

    if 'meta' not in df_initial_chunks.columns:
        df_initial_chunks['meta'] = [{} for _ in range(len(df_initial_chunks))]
    else:
        df_initial_chunks['meta'] = df_initial_chunks['meta'].apply(
            lambda x: x if isinstance(x, dict) else (ast.literal_eval(x) if isinstance(x, str) else {}))

    if not df_initial_chunks.empty:
        ner_chunks_split = np.array_split(df_initial_chunks,
                                          nb_chunk_for_parallel_ner if nb_chunk_for_parallel_ner > 0 else 1)
        print(f"[INFO - {datetime.datetime.now()}] Enriching metadata (NER) ...")
        time_ner = datetime.datetime.now()
        if torch.cuda.is_available() and nb_chunk_for_parallel_ner > 1:
            print(f"[INFO] Using GPU for NER enrichment with {len(ner_chunks_split)} parallel jobs.")
            results_ner = Parallel(n_jobs=min(nb_chunk_for_parallel_ner, os.cpu_count(), len(ner_chunks_split)))(
                delayed(enrich_df_with_ner_pipe)(chunk.copy()) for chunk in tqdm(ner_chunks_split)
            )
            results_ner = [r for r in results_ner if r is not None and not r.empty]
            if results_ner:
                df_ner_enriched = pd.concat(results_ner, ignore_index=True)
            else:
                print("[AVERTISSEMENT] Aucun résultat après l'enrichissement NER parallèle.")
                df_ner_enriched = pd.DataFrame(
                    columns=df_initial_chunks.columns if not df_initial_chunks.empty else ['text', 'meta'])
        else:
            print("[INFO] Using CPU for NER enrichment or nb_chunk_for_parallel_ner <= 1.")
            if not df_initial_chunks.empty:
                df_ner_enriched = enrich_df_with_ner_pipe(df_initial_chunks.copy())
            else:
                df_ner_enriched = pd.DataFrame(
                    columns=df_initial_chunks.columns if not df_initial_chunks.empty else ['text', 'meta'])
        print(f"[INFO - {datetime.datetime.now()}] Metadata (NER) enriched in {datetime.datetime.now() - time_ner}")
    else:
        print("[AVERTISSEMENT] df_initial_chunks est vide avant l'étape NER.")
        df_ner_enriched = pd.DataFrame(columns=['text', 'meta'] + (
            [col for col in df.columns if col not in ['text', 'meta']] if not df.empty else []))

    if not df_ner_enriched.empty and 'text' in df_ner_enriched.columns:
        df_ner_enriched.dropna(subset=['text'], inplace=True)
        df_ner_enriched = df_ner_enriched[df_ner_enriched['text'].str.strip() != '']
    elif 'text' not in df_ner_enriched.columns and not df_ner_enriched.empty:
        print("[AVERTISSEMENT] Colonne 'text' non trouvée dans df_ner_enriched.")

    if df_ner_enriched.empty:
        print("[AVERTISSEMENT] df_ner_enriched est vide après le nettoyage NER. Aucune donnée pour la génération Q/A.")
        pd.DataFrame().to_parquet(checkpoint_file_path, index=False)  # Sauvegarde d'un DF vide
        with open(checkpoint_flag_path, "w") as f:
            f.write(f"Processed at {datetime.datetime.now()}, no data after NER.")
        return df_ner_enriched

    print(
        f"[INFO - {datetime.datetime.now()}] Préparation des segments pour la génération Q/A à partir de {len(df_ner_enriched)} chunks enrichis NER...")
    segments_for_qab = []
    for index, row in df_ner_enriched.iterrows():
        meta = row.get('meta', {})
        if isinstance(meta, str):
            try:
                meta = ast.literal_eval(meta)
            except Exception as e_parse_meta:
                print(
                    f"[AVERTISSEMENT] Impossible de parser meta pour la ligne {index}: '{meta}'. Erreur: {e_parse_meta}. Utilisation de dict vide.")
                meta = {}

        current_text_segment = str(row.get('text', ''))
        current_title = str(meta.get('title', 'Titre inconnu'))
        current_entities = meta.get('entities', [])
        current_lang = meta.get('lang', 'fr')

        segments_for_qab.append({
            **row.to_dict(),
            'text_segment': current_text_segment,
            'title': current_title,
            'entities': current_entities,
            'lang': current_lang,
        })

    qab_device = str(llm_model_for_qa.device if hasattr(llm_model_for_qa, 'device') else \
                         (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))

    print(
        f"[INFO] Appel de process_segments_in_batches avec {len(segments_for_qab)} segments sur le device {qab_device}")
    processed_qab_data = process_segments_in_batches(
        segments_data=segments_for_qab,
        llm_qab_model=llm_model_for_qa,
        llm_qab_tokenizer=llm_tokenizer_for_qa,
        bert_score_model_type="bert-base-multilingual-cased",
        batch_size_llm=256,
        batch_size_bertscore=512,
        device=qab_device,
        max_new_tokens_qa=350,
        temperature_qa=0.3,
    )

    df_contextual_chunks = pd.DataFrame(processed_qab_data) if processed_qab_data else pd.DataFrame()

    expected_cols_final = list(df_ner_enriched.columns) if not df_ner_enriched.empty else (
        list(df_initial_chunks.columns) if 'df_initial_chunks' in locals() and not df_initial_chunks.empty else ['text',
                                                                                                                 'meta'])
    expected_cols_final += ['generated_question', 'generated_answer', 'llm_confidence', 'bert_score_f1',
                            'error_qab_generation', 'error_bertscore']

    current_cols = list(df_contextual_chunks.columns)
    for col in expected_cols_final:
        if col not in current_cols:
            df_contextual_chunks[col] = None
            if col in ['bert_score_f1', 'llm_confidence']:
                df_contextual_chunks[col] = 0.0

    df_contextual_chunks.to_parquet(checkpoint_file_path, index=False)
    with open(checkpoint_flag_path, "w") as f:
        f.write(f"Processed QAB at {datetime.datetime.now()}")
    debug_csv_path = os.path.join(save_dir, "debug_df_contextual_qab_chunks.csv")
    df_contextual_chunks.to_csv(debug_csv_path, index=False)
    print(f"[INFO] Chunks contextuels avec Q/A sauvegardés dans {debug_csv_path}")
    print(
        f"[INFO - {datetime.datetime.now()}] Tous les documents ont été prétraités (incl. Q/A) en {datetime.datetime.now() - t_time}")
    return df_contextual_chunks