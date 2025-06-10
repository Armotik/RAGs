import ast
import os
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
import torch
import datetime

from .qa_handler import process_segments_in_batches
# Ensure these local imports are correct based on your project structure
from .chunking import chunking
from .content_extraction import load_lang
from .text_cleaning import clean_text
from .date_extraction import extract_date
from .NER import enrich_df_with_ner_pipe


def preprocess_data(
        languages: list[str],
        max_docs_per_lang: int,
        nb_chunk_for_parallel_ner: int,
        llm_model_name_for_qa: str,  # Changed: Pass model name string
        llm_tokenizer_name_for_qa: str,  # Changed: Pass tokenizer name string
        new_docs: bool = False,
        qab_num_workers: int = -1  # New: Control parallelism for QA
) -> pd.DataFrame:
    """
    Main function to preprocess documents for Q/A generation.
    Now takes LLM model/tokenizer names and a parallelism control for QA.
    """
    t_time = datetime.datetime.now()
    save_dir = 'data'
    os.makedirs(save_dir, exist_ok=True)

    # Checkpoint paths (remain the same)
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
    results_load = Parallel(n_jobs=len(languages))(  # Parallel document loading per language
        delayed(load_lang)(lang, max_docs_per_lang) for lang in languages
    )
    all_docs = [doc for lang_docs in results_load for doc in lang_docs if lang_docs]  # Ensure lang_docs is not None
    print(f"[INFO - {datetime.datetime.now()}] Documents loaded in {datetime.datetime.now() - time_load}")

    if not all_docs:
        print("[AVERTISSEMENT] Aucun document chargé. Arrêt du prétraitement.")
        return pd.DataFrame()

    df = pd.DataFrame(all_docs)
    if df.empty:
        print("[AVERTISSEMENT] DataFrame vide après chargement des documents. Arrêt.")
        return df

    df["title"] = df["title"].apply(clean_text)
    df["text"] = df["text"].apply(clean_text)

    df_initial_chunks = chunking(df, 0)  # Assuming 0 is a valid parameter for your chunking
    if df_initial_chunks.empty:
        print("[AVERTISSEMENT] Aucun chunk initial après 'chunking'. Arrêt du prétraitement.")
        return df_initial_chunks

    print(f"[INFO - {datetime.datetime.now()}] Enriching metadata (dates) ...")
    time_date = datetime.datetime.now()
    df_initial_chunks = extract_date(df_initial_chunks)  # Assuming this modifies or returns DataFrame
    print(f"[INFO - {datetime.datetime.now()}] Metadata (dates) enriched in {datetime.datetime.now() - time_date}")

    # Initialize 'meta' column if not present, ensure it's a dict
    if 'meta' not in df_initial_chunks.columns:
        df_initial_chunks['meta'] = [{} for _ in range(len(df_initial_chunks))]
    else:
        df_initial_chunks['meta'] = df_initial_chunks['meta'].apply(
            lambda x: x if isinstance(x, dict) else (
                ast.literal_eval(x) if isinstance(x, str) and x.startswith('{') else {}))

    # NER Enrichment
    df_ner_enriched = pd.DataFrame()  # Initialize
    if not df_initial_chunks.empty:
        # Split for parallel NER processing
        ner_chunks_split = np.array_split(df_initial_chunks,
                                          nb_chunk_for_parallel_ner if 0 < nb_chunk_for_parallel_ner < len(
                                              df_initial_chunks) else 1)

        print(f"[INFO - {datetime.datetime.now()}] Enriching metadata (NER) ...")
        time_ner = datetime.datetime.now()

        # Determine NER parallelism (Joblib for NER if multiple chunks)
        # enrich_df_with_ner_pipe should handle its own device management or be CPU-bound for this parallelization
        if len(ner_chunks_split) > 1:
            print(f"[INFO] Using {len(ner_chunks_split)} parallel jobs for NER enrichment.")
            # Use min with os.cpu_count() to not oversubscribe
            n_ner_jobs = min(len(ner_chunks_split), os.cpu_count() if os.cpu_count() else 1)
            results_ner = Parallel(n_jobs=n_ner_jobs)(
                delayed(enrich_df_with_ner_pipe)(chunk.copy()) for chunk in
                tqdm(ner_chunks_split, desc="NER Processing")
            )
            results_ner = [r for r in results_ner if r is not None and not r.empty]
            if results_ner:
                df_ner_enriched = pd.concat(results_ner, ignore_index=True)
            else:
                print("[AVERTISSEMENT] Aucun résultat après l'enrichissement NER parallèle.")
        elif not df_initial_chunks.empty:  # Single chunk or no parallel NER
            print("[INFO] Using sequential NER enrichment.")
            df_ner_enriched = enrich_df_with_ner_pipe(df_initial_chunks.copy())

        print(f"[INFO - {datetime.datetime.now()}] Metadata (NER) enriched in {datetime.datetime.now() - time_ner}")
    else:
        print("[AVERTISSEMENT] df_initial_chunks est vide avant l'étape NER.")

    if df_ner_enriched.empty:  # If NER enrichment failed or yielded nothing
        print(
            "[AVERTISSEMENT] df_ner_enriched est vide après le nettoyage NER. Tentative de continuer sans NER enrichi si possible, ou arrêt.")
        # Fallback or define columns if you proceed with an empty df_ner_enriched
        # For now, if NER is crucial and yields empty, Q/A gen might not be meaningful.
        # Depending on requirements, you might use df_initial_chunks here or return empty.
        # Let's assume NER data is important.
        if df_initial_chunks.empty:  # if even initial chunks were empty.
            pd.DataFrame().to_parquet(checkpoint_file_path, index=False)
            with open(checkpoint_flag_path, "w") as f:
                f.write(f"Processed at {datetime.datetime.now()}, no data after NER.")
            return pd.DataFrame()  # Return empty if no data post NER.
        else:  # If NER failed, but initial chunks exist, you *could* proceed with initial_chunks for Q/A
            print(
                "[AVERTISSEMENT] NER failed to produce data. Q/A generation will proceed on data pre-NER if available.")
            df_ner_enriched = df_initial_chunks.copy()  # Fallback to pre-NER data
            if 'entities' not in df_ner_enriched.columns:  # Ensure 'entities' exists if qa_handler expects it
                df_ner_enriched['entities'] = [[] for _ in range(len(df_ner_enriched))]

    # Clean up df_ner_enriched (text column presence and non-empty)
    if not df_ner_enriched.empty and 'text' in df_ner_enriched.columns:
        df_ner_enriched.dropna(subset=['text'], inplace=True)
        df_ner_enriched = df_ner_enriched[df_ner_enriched['text'].str.strip() != '']
    elif 'text' not in df_ner_enriched.columns and not df_ner_enriched.empty:
        print("[AVERTISSEMENT] Colonne 'text' non trouvée dans df_ner_enriched. Q/A peut échouer.")

    if df_ner_enriched.empty:
        print("[AVERTISSEMENT] df_ner_enriched est vide. Aucune donnée pour la génération Q/A.")
        pd.DataFrame().to_parquet(checkpoint_file_path, index=False)  # Save empty checkpoint
        with open(checkpoint_flag_path, "w") as f: f.write(f"Processed at {datetime.datetime.now()}, no data for Q/A.")
        return df_ner_enriched

    print(
        f"[INFO - {datetime.datetime.now()}] Préparation des segments pour la génération Q/A à partir de {len(df_ner_enriched)} chunks...")
    segments_for_qab = []
    for index, row in df_ner_enriched.iterrows():
        meta = row.get('meta', {})
        if isinstance(meta, str):  # Ensure meta is a dict
            try:
                meta = ast.literal_eval(meta) if meta.startswith('{') else {}
            except:
                meta = {}

        # Ensure entities is a list, even if it's missing or not a list in meta
        entities_list = meta.get('entities', [])
        if not isinstance(entities_list, list):
            entities_list = []

        segments_for_qab.append({
            **row.to_dict(),  # Include all original columns from df_ner_enriched
            'text_segment': str(row.get('text', '')),  # Ensure text_segment is present
            'title': str(meta.get('title', row.get('title', 'Titre inconnu'))),  # Get title from meta or row
            'entities': entities_list,
            'lang': str(meta.get('lang', row.get('lang', 'fr')))  # Get lang from meta or row
        })

    print(
        f"[INFO] Appel de process_segments_in_batches pour {len(segments_for_qab)} segments avec qab_num_workers={qab_num_workers}")
    processed_qab_data = process_segments_in_batches(
        segments_data=segments_for_qab,
        llm_model_name=llm_model_name_for_qa,  # Pass model name
        llm_tokenizer_name=llm_tokenizer_name_for_qa,  # Pass tokenizer name
        bert_score_model_type="bert-base-multilingual-cased",  # Or make this a parameter
        batch_size_llm=256,  # Example: Adjust or make parameter (original was 256, too large for single GPU often)
        batch_size_bertscore=512,  # Example: Adjust or make parameter (original was 512)
        max_new_tokens_qa=512,  # Or make this a parameter
        temperature_qa=0.3,  # Or make this a parameter
        num_workers=qab_num_workers  # Pass parallelism control
    )

    df_contextual_chunks = pd.DataFrame(processed_qab_data) if processed_qab_data else pd.DataFrame()

    # Ensure expected columns exist in the final DataFrame
    # Base columns from df_ner_enriched or df_initial_chunks if ner failed
    base_cols = list(df_ner_enriched.columns) if not df_ner_enriched.empty else \
        (list(df_initial_chunks.columns) if 'df_initial_chunks' in locals() and not df_initial_chunks.empty else [
            'text', 'meta'])

    qa_cols = ['generated_question', 'generated_answer', 'llm_confidence', 'bert_score_f1',
               'error_qab_generation', 'error_bertscore']  # 'error_bertscore' might be useful if added in qa_handler

    expected_cols_final = base_cols + [col for col in qa_cols if col not in base_cols]

    current_cols = list(df_contextual_chunks.columns)
    for col_name in expected_cols_final:
        if col_name not in current_cols:
            df_contextual_chunks[col_name] = None  # Initialize missing columns
            if col_name in ['bert_score_f1', 'llm_confidence']:  # Default numeric for scores
                df_contextual_chunks[col_name] = 0.0
            if col_name == 'error_qab_generation' and 'error_qab_generation' not in current_cols:  # specific init
                df_contextual_chunks['error_qab_generation'] = None

    # Save results
    df_contextual_chunks.to_parquet(checkpoint_file_path, index=False)
    with open(checkpoint_flag_path, "w") as f:
        f.write(f"Processed QAB at {datetime.datetime.now()}")

    # Optional debug CSV
    debug_csv_path = os.path.join(save_dir, "debug_df_contextual_qab_chunks.csv")
    if not df_contextual_chunks.empty:
        df_contextual_chunks.to_csv(debug_csv_path, index=False)
        print(f"[INFO] Chunks contextuels avec Q/A sauvegardés dans {debug_csv_path}")
    else:
        print(f"[INFO] df_contextual_chunks est vide, rien à sauvegarder dans {debug_csv_path}")

    print(
        f"[INFO - {datetime.datetime.now()}] Tous les documents ont été prétraités (incl. Q/A) en {datetime.datetime.now() - t_time}")
    return df_contextual_chunks
