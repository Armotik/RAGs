from retrieval import multi_query_fusion
from reranking import rerank_documents

from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

model_name = "intfloat/multilingual-e5-large-instruct"
collection_name = "rag_v1"

print("[INFO] Loading model...")

model = SentenceTransformer(model_name, trust_remote_code=True)
client = MilvusClient(f"{collection_name}_milvus.db")

print("[INFO] Model loaded.")

query="Qui est Antoine Meillet ?"

# variations = [
#     "Pourquoi l’Empire romain s’est-il effondré ?",
#     "Quelles sont les causes de la disparition de l’Empire romain ?",
#     "Quels événements ont conduit à la fin de l’Empire romain ?"
# ]

docs = multi_query_fusion(
    query=query,
    collection_name=collection_name,
    client=client,
    encoder=model,
    # variations=variations,
)

print([doc['text'] for doc in docs])

reranked_docs = rerank_documents(
    query=query,
    docs=docs,
    model=model,
    top_k=10
)

print("------------------------------")
print([doc['text'] for doc in reranked_docs])