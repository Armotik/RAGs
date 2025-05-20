from typing import List, Dict

from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient


def multi_query_fusion(
        query: str,
        collection_name: str,
        client: MilvusClient,
        variations=None,
        top_k: int = 10,
        encoder: SentenceTransformer = SentenceTransformer("intfloat/multilingual-e5-large-instruct"),

) -> List[Dict]:
    """
    Perform multi-query fusion to retrieve documents from a Milvus collection.
    :param collection_name: The name of the Milvus collection to search.
    :param client: The Milvus client instance.
    :param encoder: The encoder to use for encoding queries.
    :param query: The main query string.
    :param variations: A list of variations or related queries.
    :param top_k: The number of top documents to retrieve.
    :return: A list of dictionaries containing document IDs, texts, and scores.
    """

    # TODO : Générer une liste de variations à partir de la requête principale si ce n'est pas fourni

    if variations is None:
        variations = []

    all_queries = [query] + variations
    query_vectors = encoder.encode(all_queries, normalize_embeddings=True)

    doc_scores = {}
    doc_texts = {}
    doc_vectors = {}

    for vec in query_vectors:
        results = client.search(
            collection_name=collection_name,
            data=[vec.tolist()],
            anns_field="vector",
            search_params={"metric_type": "COSINE"},
            limit=top_k,
            output_fields=["text", "vector"]
        )[0]

        for hit in results:
            doc_id = hit["id"]
            text = hit["entity"]["text"]
            score = 1 - hit["distance"]

            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
            doc_texts[doc_id] = text
            doc_vectors[doc_id] = hit["entity"].get("vector")

    for doc_id in doc_scores:
        doc_scores[doc_id] /= len(all_queries)

    sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k]
    results = []
    for doc_id, score in sorted_docs:
        results.append({
            "id": doc_id,
            "text": doc_texts[doc_id],
            "score": round(score, 4),
            "vector": doc_vectors[doc_id]  # Optional
        })

    return results
