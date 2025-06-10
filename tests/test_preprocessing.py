import sys
import os
import pandas as pd
import torch
from mpmath import mp

# Ajout du chemin racine pour que les imports de RAGs.preprocessing fonctionnent
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# On importe uniquement la fonction principale de prétraitement
from preprocessing.pretraitment import preprocess_data


def run_pipeline_test():
    """Lance le pipeline de prétraitement complet avec des paramètres de test."""
    print("=" * 60)
    print("[TEST SCRIPT] Lancement du pipeline de test complet...")
    print("=" * 60)

    # --- Configuration ---
    # Utiliser des valeurs faibles pour un test rapide
    config = {
        "languages": ['fr'],
        "max_docs_per_lang": 10,
        "nb_chunk_for_parallel_ner": 2,
        "llm_model_name_for_qa": "mistralai/Mistral-7B-Instruct-v0.3",  # Modèle performant
        "new_docs": True,  # Forcer le recalcul pour chaque test
    }

    print("\nConfiguration utilisée pour ce test :")
    for key, value in config.items():
        print(f"  - {key}: {value}")
    print("-" * 30)

    # --- Exécution du pipeline ---
    res_df = preprocess_data(
        languages=config["languages"],
        max_docs_per_lang=config["max_docs_per_lang"],
        nb_chunk_for_parallel_ner=config["nb_chunk_for_parallel_ner"],
        llm_model_name_for_qa=config["llm_model_name_for_qa"],
        new_docs=config["new_docs"],
    )

    # --- Affichage et Vérification des résultats ---
    if not res_df.empty:
        print("\n[TEST SCRIPT] Le pipeline a retourné un DataFrame non vide.")
        print(f"Nombre total de lignes traitées : {len(res_df)}")

        # Afficher les 5 premières lignes pour un aperçu
        print("\n--- APERÇU DES RÉSULTATS (5 premières lignes) ---")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(res_df.head())
        print("-" * 30)

        # Vérifier la présence des colonnes générées
        print("\n[TEST SCRIPT] Vérification des colonnes attendues...")
        required_cols = ['text', 'meta', 'generated_question', 'generated_answer']
        missing_cols = [col for col in required_cols if col not in res_df.columns]

        if not missing_cols:
            print("[SUCCESS] Toutes les colonnes principales sont présentes.")
            valid_qa_count = res_df['generated_question'].notna().sum()
            print(f"Nombre de questions générées avec succès : {valid_qa_count} / {len(res_df)}")
        else:
            print(f"[FAILURE] Colonnes manquantes : {missing_cols}")

        output_csv_path = "test_pipeline_output.csv"
        res_df.to_csv(output_csv_path, index=False)
        print(f"\n[TEST SCRIPT] Résultats complets du test sauvegardés dans : {output_csv_path}")

    else:
        print("\n[FAILURE] Le pipeline a retourné un DataFrame vide.")

    print("\n[TEST SCRIPT] Fin du test.")
    print("=" * 60)


if __name__ == "__main__":
    # Nécessaire pour la compatibilité de multiprocessing avec CUDA
    if "spawn" not in mp.get_start_method(allow_none=True):
        mp.set_start_method("spawn", force=True)
    run_pipeline_test()