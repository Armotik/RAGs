from pymilvus import MilvusClient, Collection, Connections

client = MilvusClient("../rag_v1_milvus.db")
collection_name = "rag_v1"

print(client.describe_collection(collection_name))

collection = Collection(collection_name)

print(collection.num_entities)