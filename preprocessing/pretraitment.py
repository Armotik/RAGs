import os

import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
import torch

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

    if os.path.exists("../checkpoint__df_chunk_ready.json") and not new_docs:
        print("[INFO] Chargement des données déjà prétraitées...")
        df_chunk = pd.read_parquet("df_chunk_ready.parquet")

        return df_chunk

    else:

        print("Loading documents...")

        results = Parallel(n_jobs=len(languages))(
            delayed(load_lang)(lang, max_docs_per_lang) for lang in languages
        )

        all_docs = [doc for lang_docs in results for doc in lang_docs]

        print("Documents loaded.")

        df = pd.DataFrame(all_docs)
        df["title"] = df["title"].apply(clean_text)
        df["text"] = df["text"].apply(clean_text)

        df_chunk = chunking(df, 0)

        print("Enriching metadata (dates) ...")

        df_chunk = extract_date(df_chunk)

        chunks = np.array_split(df_chunk, nb_chunk)

        print("Enriching metadata (NER) ...")



        if not torch.cuda.is_available():
            print("Using GPU for NER enrichment.")
            results = Parallel(n_jobs=8)(
                delayed(enrich_df_with_ner_pipe)(chunk) for chunk in tqdm(chunks)
            )

            df_chunk = pd.concat(results, ignore_index=True)
        else:
            print("Using CPU for NER enrichment.")
            results = enrich_df_with_ner_pipe(df_chunk)

            df_chunk = results

        df_chunk.dropna(inplace=True)

        df_chunk.to_parquet("../data/df_chunk_ready.parquet")
        with open("../data/checkpoint__df_chunk_ready.json", "w") as f:
            f.write("done")

        return df_chunk