from transformers import AutoTokenizer, AutoModelForCausalLM

from retrieval import multi_query_fusion
from reranking import rerank_documents
from generation import generate_response

from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import torch
import os
from dotenv import load_dotenv

model_name = "intfloat/multilingual-e5-large-instruct"
collection_name = "rag_v1"

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

print("[INFO] Loading model...")

model = SentenceTransformer(model_name, trust_remote_code=True)
client = MilvusClient(f"{collection_name}_milvus.db")

print("[INFO] Model loaded.")

query = "Quelles ont été les conséquences majeures de la Première Guerre mondiale en Europe ?"

variations = [

    "Comment la Première Guerre mondiale a-t-elle transformé la société européenne ?",

    "Quels changements politiques et sociaux l'Europe a-t-elle connus après 1918 ?",

    "Impact de la Grande Guerre sur les frontières et les nations en Europe.",

    "Décrivez les principaux bouleversements économiques en Europe suite à la guerre de 14-18.",

    "Quelles étaient les répercussions à long terme du premier conflit mondial sur le continent européen ?"

]

docs = multi_query_fusion(
    query=query,
    collection_name=collection_name,
    client=client,
    encoder=model,
    variations=variations,
    top_k=20
)

reranked_docs = rerank_documents(
    query=query,
    docs=docs,
    model=model,
    top_k=10
)

llm_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Chargement du LLM: {llm_model_name} sur {device}...")

llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN,
)

if llm_tokenizer.pad_token_id is None:
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

print("[INFO] LLM chargé.")

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
