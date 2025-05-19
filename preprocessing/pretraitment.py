import os

import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
import torch
import datetime

from .chunking import chunking
from .content_extraction import load_lang
from .text_cleaning import clean_text
from .date_extraction import extract_date
from .NER import enrich_df_with_ner_pipe

def preprocess_data(languages: list[str], max_docs_per_lang: int, nb_chunk:int, new_docs=False) -> pd.DataFrame:
    """
    Preprocessing logic
    :param nb_chunk: number of chunks to split the dataframe into
    :param languages: the list of languages to load
    :param max_docs_per_lang: maximum number of documents to load per language
    :param new_docs: if True, load new documents
    :return: a pandas DataFrame containing all documents
    """

    t_time = datetime.datetime.now()

    if os.path.exists("data/checkpoint__df_chunk_ready.json") and not new_docs:
        print("[INFO] Loading documents from checkpoint...")
        df_chunk = pd.read_parquet("data/df_chunk_ready.parquet")

        print(f"[INFO - {datetime.datetime.now()}] Documents loaded in {datetime.datetime.now() - t_time}")

        return df_chunk

    else:

        print(f"[INFO - {datetime.datetime.now()}] Loading documents...")
        time = datetime.datetime.now()

        results = Parallel(n_jobs=len(languages))(
            delayed(load_lang)(lang, max_docs_per_lang) for lang in languages
        )

        all_docs = [doc for lang_docs in results for doc in lang_docs]

        print(f"[INFO - {datetime.datetime.now()}] Documents loaded in {datetime.datetime.now() - time}")

        df = pd.DataFrame(all_docs)
        df["title"] = df["title"].apply(clean_text)
        df["text"] = df["text"].apply(clean_text)

        df_chunk = chunking(df, 0)

        print(f"[INFO - {datetime.datetime.now()}] Enriching metadata (dates) ...")
        time = datetime.datetime.now()

        df_chunk = extract_date(df_chunk)

        print(f"[INFO - {datetime.datetime.now()}] Metadata enriched in {datetime.datetime.now() - time}")

        chunks = np.array_split(df_chunk, nb_chunk)

        print(f"[INFO - {datetime.datetime.now()}] Enriching metadata (NER) ...")
        time = datetime.datetime.now()

        if torch.cuda.is_available():
            print("[INFO] Using GPU for NER enrichment.")
            results = Parallel(n_jobs=8)(
                delayed(enrich_df_with_ner_pipe)(chunk) for chunk in tqdm(chunks)
            )

            df_chunk = pd.concat(results, ignore_index=True)
        else:
            print("[INFO] Using CPU for NER enrichment.")
            results = enrich_df_with_ner_pipe(df_chunk)

            df_chunk = results

        print(f"[INFO - {datetime.datetime.now()}] Metadata enriched in {datetime.datetime.now() - time}")

        df_chunk.dropna(inplace=True)

        save_dir = 'data'
        os.makedirs(save_dir, exist_ok=True)

        df_chunk.to_parquet(os.path.join(save_dir, "df_chunk_ready.parquet"), index=False)
        with open("checkpoint__df_chunk_ready.json", "w") as f:
            f.write("done")

        print(f"[INFO - {datetime.datetime.now()}] Documents preprocessed in {datetime.datetime.now() - t_time}")
        return df_chunk