import torch
from pymilvus import MilvusClient
from transformers import AutoTokenizer, AutoModelForCausalLM

from preprocessing import preprocess_data
from embedding import vectorisation
from indexation import database_creation, indexation

###
###
###

print("[INFO] - Starting the application...")

languages = ['fr', 'en']
max_docs_per_lang = 100 # taille du dataset (x2 car français + anglais)

llm_model_name_for_qa = "mistralai/Mistral-7B-Instruct-v0.3"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Loading LLM for Q/A generation: {llm_model_name_for_qa} on {device}...")

llm_tokenizer_qab = AutoTokenizer.from_pretrained(llm_model_name_for_qa)
llm_model_qab = AutoModelForCausalLM.from_pretrained(
    llm_model_name_for_qa,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

if llm_tokenizer_qab.pad_token_id is None:
    llm_tokenizer_qab.pad_token_id = llm_tokenizer_qab.eos_token_id
print("[INFO] LLM for Q/A generation loaded.")

res = preprocess_data(
    languages,
    max_docs_per_lang,
    8,
    llm_model_qab,
    llm_tokenizer_qab,
    new_docs=True,
)

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