import numpy as np
import pandas as pd
from pymilvus import MilvusClient
from tqdm import tqdm


def indexation(df:pd.DataFrame, embeddings:np.ndarray, client:MilvusClient, collection_name:str) -> None:
    """
    Indexation of the data in the database.
    :param collection_name: the name of the collection to index the data in.
    :param embeddings: the embeddings to index.
    :param df: the dataframe to index.
    :param client: the client to use to index the data.
    :return: None
    """

    data = []

    for i, row in df.iterrows():
        entry = {
            "id": i,
            "vector": embeddings[i],
            "text": row["text"],
            "title": row["meta"]["title"],
            "lang": row["meta"]["lang"],
            "dates_iso": row["meta"]["dates_iso"],
            "earliest_date": row["meta"]["earliest_date"],
            "latest_date": row["meta"]["latest_date"],
            "entities": [ent["text"] for ent in row["meta"].get("entities", [])],
        }

        data.append(entry)

    batch_size = 1000

    for i in tqdm(range(0, len(data), batch_size), desc="Insertion dans Milvus"):
        batch = data[i:i + batch_size]
        client.insert(collection_name=collection_name, data=batch)