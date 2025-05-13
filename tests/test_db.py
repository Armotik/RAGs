from pymilvus import MilvusClient

client = MilvusClient("../test_milvus.db")
collection_name = "test"

print(client.describe_collection(collection_name))

offset = 0
batch_size = 1000

while True:
    results = client.query(
        collection_name=collection_name,
        filter=None,
        output_fields=[
            "id", "text", "title", "lang", "docid",
            "dates", "entities"
        ],
        limit=batch_size,
        offset=offset
    )

    if not results:
        break

    for r in results:
        print(r)

    offset += batch_size