from typing import Literal

import pandas as pd
import numpy as np
from numpy import ndarray
from sentence_transformers import SentenceTransformer

def vectorisation(df: pd.DataFrame, model_name:str, backend:Literal["torch", "onnx", "openvino"], batch_size:int) -> np.ndarray:
    """
    Vectorises the text data in the dataframe using the specified model and backend.
    :param batch_size: the batch size to use for vectorisation
    :param model_name: the name of the model to use for vectorisation
    :param backend: the backend to use for vectorisation (e.g. 'onnx', 'torch')
    :param df: the dataframe containing the text data to vectorise
    :return: a numpy array of the vectorised text data
    """

    embedding_model = SentenceTransformer(model_name, trust_remote_code=True, backend=backend)

    df["chunk"] = df["meta"].apply(lambda m: m.get("title", "")) + "\n" + df["text"]

    texts = df["chunk"].tolist()

    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return np.array(embeddings)

def query_vectorisation(
    query: str,
    model_name:str,
    backend:Literal["torch", "onnx", "openvino"],
    batch_size:int
) -> ndarray:
    """
    Vectorises the query using the specified model and backend.
    :param batch_size: the batch size to use for vectorisation
    :param model_name: the name of the model to use for vectorisation
    :param backend: the backend to use for vectorisation (e.g. 'onnx', 'torch')
    :param query: the query to vectorise
    :return: a numpy array of the vectorised query
    """

    embedding_model = SentenceTransformer(model_name, trust_remote_code=True, backend=backend)

    embeddings = embedding_model.encode(
        [query],
        batch_size=batch_size,
        normalize_embeddings=True
    )

    return embeddings