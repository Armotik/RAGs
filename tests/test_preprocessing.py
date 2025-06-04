# tests/test_preprocessing.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.pretraitment import preprocess_data

print("[INFO] - Starting the application...")

languages = ['fr', 'en']
max_docs_per_lang = 1000
nb_chunk_for_parallel_ner = 20

llm_model_name_for_qa = "mistralai/Mistral-7B-Instruct-v0.3"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(
    f"[INFO] Loading LLM for Q/A generation: {llm_model_name_for_qa} on device_map='auto' (effective primary device: {device})...")

llm_tokenizer_qab = AutoTokenizer.from_pretrained(llm_model_name_for_qa)  # , token=hf_token)

if llm_tokenizer_qab.pad_token_id is None:
    print("[INFO] Setting pad_token_id to eos_token_id for Q/A tokenizer.")
    llm_tokenizer_qab.pad_token_id = llm_tokenizer_qab.eos_token_id
llm_tokenizer_qab.padding_side = "left"
print(f"[INFO] Q/A tokenizer padding side set to: {llm_tokenizer_qab.padding_side}")

llm_model_qab = AutoModelForCausalLM.from_pretrained(
    llm_model_name_for_qa,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
print("[INFO] LLM for Q/A generation loaded.")

res_df = preprocess_data(
    languages,
    max_docs_per_lang,
    nb_chunk_for_parallel_ner,
    llm_model_qab,
    llm_tokenizer_qab,
    new_docs=True,
)

print(f"\n[INFO] Contenu de res_df après preprocess_data (premières 5 lignes) :")
if not res_df.empty:
    print(res_df.head())
    print(f"\n[INFO] Colonnes de res_df: {res_df.columns.tolist()}")
    print(f"[INFO] Nombre de lignes dans res_df: {len(res_df)}")

    if 'generated_question' in res_df.columns and 'generated_answer' in res_df.columns:
        print("\n[INFO] Aperçu des Q/A générées (premiers non-nuls) :")
        valid_qa = res_df[res_df['generated_question'].notna() & res_df['generated_answer'].notna()]
        if not valid_qa.empty:
            for index, row in valid_qa.head().iterrows():  # Affiche jusqu'à 5 Q/A valides
                print(f"--- Entrée {index} ---")
                print(f"  Texte Original (début): {str(row.get('text_segment', row.get('text', ''))[:150])}...")
                print(f"  Q: {row['generated_question']}")
                print(f"  A: {row['generated_answer']}")
                print(f"  LLM Conf: {row.get('llm_confidence', 'N/A')}, BERT F1: {row.get('bert_score_f1', 'N/A')}")
        else:
            print("Aucune Q/A valide (non-nulle pour Q et A) n'a été générée dans l'échantillon traité.")

        print(
            f"\n[INFO] Nombre total de Q/A valides (question ET réponse non nulles): {len(valid_qa)} sur {len(res_df)} segments traités.")

    else:
        print(
            "[ERREUR] Les colonnes 'generated_question' ou 'generated_answer' sont manquantes dans le DataFrame final.")
else:
    print("[INFO] Le DataFrame résultant (res_df) est vide.")

output_csv_path = "test_preprocessing_output_qab.csv"
if not res_df.empty:
    res_df.to_csv(output_csv_path, index=False)
    print(f"\n[INFO] Résultats complets sauvegardés dans : {output_csv_path}")
else:
    print(f"[INFO] Aucun résultat à sauvegarder dans {output_csv_path} car le DataFrame est vide.")