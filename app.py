from pymilvus import MilvusClient

from preprocessing import preprocess_data
from embedding import vectorisation, query_vectorisation
from indexation import database_creation, indexation, test_search

###
###
###

languages = ['fr', 'en']
max_docs_per_lang = 2000000 # taille du dataset (x2 car français + anglais)

res = preprocess_data(languages, max_docs_per_lang, 400)

embeddings = vectorisation(
    res,
    "intfloat/multilingual-e5-large-instruct",
    'torch',
    128
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
    drop_collection=False,
    dim=embeddings.shape[1],
    index_param=index_params,
)

indexation(
    df=res,
    embeddings=embeddings,
    client=client,
    collection_name="v1",
)

query = "Qui est Antoin Meillet ?"

query_vector = query_vectorisation(
    query,
    "intfloat/multilingual-e5-large-instruct",
    'torch',
    16
)

results = test_search(
    client=client,
    query_vector=query_vector,
    collection_name="v1",
    top_k=1
)