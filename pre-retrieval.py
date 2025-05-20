from pymilvus import MilvusClient

from preprocessing import preprocess_data
from embedding import vectorisation
from indexation import database_creation, indexation

###
###
###

print("[INFO] - Starting the application...")

languages = ['fr', 'en']
max_docs_per_lang = 50000 # taille du dataset (x2 car français + anglais)

res = preprocess_data(languages, max_docs_per_lang, 200)

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
    name="rag_v1",
    drop_collection=True,
    dim=embeddings.shape[1],
    index_param=index_params,
)

indexation(
    df=res,
    embeddings=embeddings,
    client=client,
    collection_name="rag_v1",
)