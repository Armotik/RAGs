from pymilvus import MilvusClient

from preprocessing import preprocess_data
from embedding import vectorisation, query_vectorisation
from indexation import database_creation, indexation, test_search

###
###
###

languages = ['fr', 'en']
max_docs_per_lang = 100 # taille du dataset (x2 car français + anglais)

res = preprocess_data(languages, max_docs_per_lang, 64)

embeddings = vectorisation(
    res,
    "intfloat/e5-small-v2",
    'onnx',
    16
)

index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    index_name="vector_index",
    metric_type="cosine",
    params={
        "M": 48,
        "efConstruction": 200,
    },
)

client = database_creation.create_database(
    name="test",
    drop_collection=True,
    dim=embeddings.shape[1],
    index_param=index_params,
)

indexation(
    df=res,
    embeddings=embeddings,
    client=client,
    collection_name="test",
)

query = "Qui est Antoin Meillet ?"

query_vector = query_vectorisation(
    query,
    "intfloat/e5-small-v2",
    'onnx',
    16
)

results = test_search(
    client=client,
    query_vector=query_vector,
    collection_name="test",
    top_k=1
)

# data: ["[{'id': 1, 'distance': 0.8859543204307556, 'entity': {'text': 'Il est aussi philologue.', 'title': 'Antoine Meillet', 'lang': 'fr'}}]"]
# data: ["[{'id': 1, 'distance': 0.8859543204307556, 'entity': {}}]"]


print(results)