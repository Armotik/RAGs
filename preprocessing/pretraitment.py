import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
import datetime
import ast

# Importation des modules du projet
from .chunking import chunking
from .content_extraction import load_lang
from .text_cleaning import clean_text
from .date_extraction import extract_date
from .NER import enrich_df_with_ner_pipe
from .qa_handler import process_segments_for_qa


def qa_worker(segment_subset, gpu_id, model_id, batch_size):
    """
    Fonction wrapper pour appeler le traitement QA sur un sous-ensemble de données.
    """
    return process_segments_for_qa(
        segments_data=segment_subset,
        model_id=model_id,
        device=f"cuda:{gpu_id}",
    )


def preprocess_data(
        languages: list[str],
        max_docs_per_lang: int,
        nb_chunk_for_parallel_ner: int,
        llm_model_name_for_qa: str,
        new_docs=False
) -> pd.DataFrame:
    """
    Pipeline de prétraitement principal, avec parallélisation de la génération Q/A.
    """
    t_time = datetime.datetime.now()
    save_dir = 'data'
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_file_path = os.path.join(save_dir, "df_full_pipeline_output.parquet")
    checkpoint_flag_path = os.path.join(save_dir, "checkpoint_full_pipeline.json")

    if os.path.exists(checkpoint_flag_path) and not new_docs:
        print(f"[INFO] Chargement des données entièrement traitées depuis : {checkpoint_file_path}")
        return pd.read_parquet(checkpoint_file_path)

    print("[INFO] Lancement du prétraitement complet (pas de checkpoint valide ou new_docs=True).")

    # --- Étape 1: Chargement et Nettoyage ---
    print(f"[INFO] Étape 1: Chargement et nettoyage des documents...")
    time_load = datetime.datetime.now()
    results_load = Parallel(n_jobs=len(languages))(delayed(load_lang)(lang, max_docs_per_lang) for lang in languages)
    df = pd.DataFrame([doc for lang_docs in results_load for doc in lang_docs])
    if df.empty: return df
    df["title"] = df["title"].apply(clean_text)
    df["text"] = df["text"].apply(clean_text)
    print(f"[INFO] Chargement et nettoyage terminés en {datetime.datetime.now() - time_load}")

    # --- Étape 2: Chunking et Extraction de Dates ---
    print(f"[INFO] Étape 2: Chunking et extraction de dates...")
    df_chunks = chunking(df, 0)
    if df_chunks.empty: return df_chunks
    df_chunks = extract_date(df_chunks)

    # --- Étape 3: Extraction d'Entités (NER) en parallèle ---
    print(f"[INFO] Étape 3: Enrichissement NER en parallèle...")
    ner_chunks_split = np.array_split(df_chunks, nb_chunk_for_parallel_ner if nb_chunk_for_parallel_ner > 0 else 1)
    results_ner = Parallel(n_jobs=min(nb_chunk_for_parallel_ner, os.cpu_count()))(
        delayed(enrich_df_with_ner_pipe)(chunk.copy()) for chunk in tqdm(ner_chunks_split, desc="NER Processing")
    )
    df_ner_enriched = pd.concat([r for r in results_ner if r is not None and not r.empty], ignore_index=True).dropna(
        subset=['text'])

    if df_ner_enriched.empty:
        print("[AVERTISSEMENT] DataFrame vide après l'étape NER. Arrêt.")
        return pd.DataFrame()

    # --- Étape 4: Génération de Q/A en parallèle sur les GPUs ---
    print(f"\n[INFO] Étape 4: Lancement de la génération Q/A en parallèle...")
    time_qa = datetime.datetime.now()
    segments_for_qab = [row.to_dict() for _, row in df_ner_enriched.iterrows()]

    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        print(f"[INFO] {num_gpus} GPU(s) détecté(s). Distribution des {len(segments_for_qab)} segments.")
        segment_chunks_per_gpu = np.array_split(segments_for_qab, num_gpus)
        batch_size_per_gpu = 16

        pool_args = [(chunks.tolist(), i, llm_model_name_for_qa, batch_size_per_gpu) for i, chunks in
                     enumerate(segment_chunks_per_gpu) if len(chunks) > 0]

        with mp.get_context("spawn").Pool(processes=num_gpus) as pool:
            results_list = list(
                tqdm(pool.starmap(qa_worker, pool_args), total=len(pool_args), desc="Generating QA on GPUs"))

        processed_qab_data = [item for sublist in results_list for item in sublist]
    else:
        print("[INFO] Aucun GPU détecté. Traitement sur CPU (très lent)...")
        processed_qab_data = process_segments_for_qa(segments_for_qab, llm_model_name_for_qa, device="cpu")

    print(f"[INFO] Génération Q/A terminée en {datetime.datetime.now() - time_qa}")

    df_final = pd.DataFrame(processed_qab_data)

    # --- Étape 5: Sauvegarde Finale ---
    print(f"[INFO] Étape 5: Sauvegarde des résultats finaux...")
    df_final.to_parquet(checkpoint_file_path, index=False)
    with open(checkpoint_flag_path, "w") as f:
        f.write(f"Processed QAB at {datetime.datetime.now()}")
    print(f"[INFO] Pipeline complet terminé. Résultats sauvegardés dans {checkpoint_file_path}")
    print(f"Temps total d'exécution : {datetime.datetime.now() - t_time}")

    return df_final