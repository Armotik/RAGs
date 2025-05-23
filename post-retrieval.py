from transformers import AutoTokenizer, AutoModelForCausalLM

from retrieval import multi_query_fusion
from reranking import rerank_documents
from generation import generate_response

from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import torch
from dotenv import load_dotenv

model_name = "intfloat/multilingual-e5-large-instruct"
collection_name = "rag_v1"

load_dotenv()

print("[INFO] Loading models...")

model = SentenceTransformer(model_name, trust_remote_code=True)
client = MilvusClient(f"{collection_name}_milvus.db")

llm_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Load LLM : {llm_model_name} on {device}...")

llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

if llm_tokenizer.pad_token_id is None:
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

print("[INFO] LLM loaded.")

print("[INFO] Models loaded.")

query = "Expliquez en détail comment les voyages interstellaires des Romains ont influencé l'architecture des temples égyptiens"

docs = multi_query_fusion(
    query=query,
    collection_name=collection_name,
    client=client,
    encoder=model,
    llm_variation_model=llm_model,
    llm_variation_tokenizer=llm_tokenizer,
    num_variations_to_generate=4,
    generate_variations_flag=True,
    top_k=20
)

reranked_docs = rerank_documents(
    query=query,
    docs=docs,
    model=model,
    top_k=10
)

final_answer = generate_response(
    query=query,
    context_docs=reranked_docs,
    llm_model=llm_model,
    llm_tokenizer=llm_tokenizer,
    max_new_tokens=2000
)

print("\n===== Réponse Finale du LLM =====")
print(final_answer)

print("===================================")
print([doc['text'] for doc in reranked_docs])
print("===================================")
