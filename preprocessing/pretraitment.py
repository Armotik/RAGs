import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm

from .chunking import chunking
from .content_extraction import load_lang
from .text_cleaning import clean_text
from .date_extraction import parallel_enrich_meta
from .NER import enrich_df_with_ner_pipe

def preprocess_data(languages: list[str], max_docs_per_lang: int, nb_chunk:int) -> pd.DataFrame:
    """
    Preprocessing logic
    :param nb_chunk: number of chunks to split the dataframe into
    :param languages: the list of languages to load
    :param max_docs_per_lang: maximum number of documents to load per language
    :return: a pandas DataFrame containing all documents
    """

    results = Parallel(n_jobs=len(languages))(
        delayed(load_lang)(lang, max_docs_per_lang) for lang in languages
    )

    all_docs = [doc for lang_docs in results for doc in lang_docs]

    df = pd.DataFrame(all_docs)
    df["title"] = df["title"].apply(clean_text)
    df["text"] = df["text"].apply(clean_text)

    df_chunk = chunking(df, 0)

    df_chunk = parallel_enrich_meta(df_chunk)

    chunks = np.array_split(df_chunk, nb_chunk)

    results = Parallel(n_jobs=8, backend="loky", prefer="processes")(
        delayed(enrich_df_with_ner_pipe)(chunk) for chunk in tqdm(chunks)
    )

    df_chunk = pd.concat(results, ignore_index=True)

    return df_chunk