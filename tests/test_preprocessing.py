import torch
# Removed: AutoTokenizer, AutoModelForCausalLM (now loaded by workers)
import sys
import os
import pandas as pd  # For DataFrame checks

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.pretraitment import preprocess_data

print("[INFO] - Starting the application...")

languages = ['fr', 'en']  # Example languages
max_docs_per_lang = 1000  # Example: Reduced for faster testing if needed
nb_chunk_for_parallel_ner = 20  # Example: Number of chunks for NER parallelism

llm_model_name_for_qa = "mistralai/Mistral-7B-Instruct-v0.3"  # Model name string

# Device info for general context, QA part now handles its own device management per worker
# This 'device' is not directly passed to QA generation anymore.
# It can be used for other parts of your test script if needed.
primary_device_info = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Primary device available to this main script: {primary_device_info}.")
print(f"[INFO] LLM for Q/A generation will be: {llm_model_name_for_qa}")
print(f"[INFO] QA generation will use device_map='auto' within workers on their assigned GPUs/CPU.")

# LLM and Tokenizer are no longer loaded here; names are passed to preprocess_data
# Workers in qa_handler will load them.

# Call preprocess_data with model names and new qab_num_workers parameter
# qab_num_workers:
#   -1: use all available GPUs
#    0: use CPU (1 worker)
#   >0: use specified number of GPUs (capped by availability)
qab_processing_workers = -1  # Example: Use all available GPUs

print(f"[INFO] Starting preprocess_data with QAB num_workers = {qab_processing_workers}...")
res_df = preprocess_data(
    languages=languages,
    max_docs_per_lang=max_docs_per_lang,
    nb_chunk_for_parallel_ner=nb_chunk_for_parallel_ner,
    llm_model_name_for_qa=llm_model_name_for_qa,  # Pass model name
    llm_tokenizer_name_for_qa=llm_model_name_for_qa,  # Pass tokenizer name (often same as model)
    new_docs=True,  # Force reprocessing for test
    qab_num_workers=qab_processing_workers  # Control QA parallelism
)

print(f"\n[INFO] Contenu de res_df après preprocess_data (premières 5 lignes) :")
if res_df is not None and not res_df.empty:
    print(res_df.head())
    print(f"\n[INFO] Colonnes de res_df: {res_df.columns.tolist()}")
    print(f"[INFO] Nombre de lignes dans res_df: {len(res_df)}")

    if 'generated_question' in res_df.columns and 'generated_answer' in res_df.columns:
        print("\n[INFO] Aperçu des Q/A générées (premiers non-nuls) :")
        # Filter for rows where both question and answer are not None and not empty strings
        valid_qa = res_df[
            res_df['generated_question'].notna() & (res_df['generated_question'] != '') &
            res_df['generated_answer'].notna() & (res_df['generated_answer'] != '')
            ]
        if not valid_qa.empty:
            for index, row in valid_qa.head().iterrows():
                print(f"--- Entrée {index} (Original Doc ID: {row.get('original_doc_id', 'N/A')}) ---")
                # Displaying 'text_segment' if available, otherwise 'text'
                original_text_key = 'text_segment' if 'text_segment' in row else 'text'
                print(f"  Texte Original (début): {str(row.get(original_text_key, ''))[:150]}...")
                print(f"  Q: {row['generated_question']}")
                print(f"  A: {row['generated_answer']}")
                print(f"  LLM Conf: {row.get('llm_confidence', 'N/A')}, BERT F1: {row.get('bert_score_f1', 'N/A')}")
                if 'error_qab_generation' in row and pd.notna(row['error_qab_generation']):
                    print(f"  Erreur QAB: {row['error_qab_generation']}")
        else:
            print("Aucune Q/A valide (question ET réponse non nulles/vides) n'a été générée dans l'échantillon traité.")

        total_valid_qa_count = len(valid_qa)
        print(f"\n[INFO] Nombre total de Q/A valides: {total_valid_qa_count} sur {len(res_df)} segments traités.")

        # Count segments with errors during QAB generation
        if 'error_qab_generation' in res_df.columns:
            error_count = res_df['error_qab_generation'].notna().sum()
            print(f"[INFO] Nombre de segments avec erreur durant la génération Q/A: {error_count}")

    else:
        print(
            "[ERREUR] Les colonnes 'generated_question' ou 'generated_answer' sont manquantes dans le DataFrame final.")
elif res_df is None:
    print("[ERREUR] Le DataFrame résultant (res_df) est None.")
else:  # res_df is empty
    print("[INFO] Le DataFrame résultant (res_df) est vide.")

output_csv_path = "test_preprocessing_output_qab.csv"
if res_df is not None and not res_df.empty:
    try:
        res_df.to_csv(output_csv_path, index=False)
        print(f"\n[INFO] Résultats complets sauvegardés dans : {output_csv_path}")
    except Exception as e_csv:
        print(f"[ERREUR] Échec de la sauvegarde CSV dans {output_csv_path}: {e_csv}")
else:
    print(f"[INFO] Aucun résultat à sauvegarder dans {output_csv_path} car le DataFrame est vide ou None.")

print("[INFO] - Application terminée.")
