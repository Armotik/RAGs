import numpy as np
from pymilvus import MilvusClient


def test_search(query_vector, client:MilvusClient, collection_name:str, top_k:int=5) -> list[
    list[dict]]:
    """
    Test the search function of the Milvus client.
    :param query_vector: query vector to search for.
    :param top_k: number of top results to return.
    :param client: Milvus client to use to search the data.
    :return: None
    """

    results = client.search(
        collection_name=collection_name,
        data=query_vector,
        limit=top_k,
        output_fields=["text", "title", "lang"],
        params={
            "metric_type": "COSINE",
            "params": {
                "ef": 64
            }
        },
    )

    return results