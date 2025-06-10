import pandas as pd
import torch
import torch.multiprocessing as mp
import numpy as np
import time
import os
import evaluate
from tqdm import tqdm
import ast
from datasets import load_dataset
from itertools import islice
from joblib import Parallel, delayed
import unicodedata
import re
from dateparser.search import search_dates
from dateparser_data.settings import default_parsers
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM, pipeline

tqdm.pandas(desc="Processing text")

# Configuration de l'environnement pour Matplotlib (bonne pratique)
matplotlib_cache_dir = './tmp/matplotlib_cache_amudet01'
os.makedirs(matplotlib_cache_dir, exist_ok=True)
os.environ['MPLCONFIGDIR'] = matplotlib_cache_dir

# Vider le cache CUDA au début
torch.cuda.empty_cache()

# ==============================================================================
# SÉLECTION DES MODÈLES QA À BENCHMARKER
# C'est ici que vous pouvez configurer les modèles à tester.
# ==============================================================================
MODELS_TO_BENCHMARK = [
    {"model_id": "valhalla/t5-small-qg-hl", "size_category": "small"},
    {"model_id": "mrm8488/mT5-small-finetuned-tydiqa-for-xqa", "size_category": "small"},
    {"model_id": "valhalla/t5-base-qg-hl", "size_category": "medium"},
    {"model_id": "valhalla/t5-large-qg-hl", "size_category": "large"},
    {"model_id": "google/flan-t5-large", "size_category": "large"},
]


# ==============================================================================
# SECTION 1: FONCTIONS DU PIPELINE DE PRÉTRAITEMENT COMPLET
# ==============================================================================

def load_lang(lang: str, max_docs: int) -> list:
    """
    Charge le dataset pour une langue spécifique et retourne une liste de documents.
    """
    print(f"[{lang.upper()}] Chargement...")
    dataset_stream = load_dataset('miracl/miracl-corpus', lang, split='train', streaming=True, trust_remote_code=True)
    sample = islice(dataset_stream, max_docs) if max_docs > 0 else dataset_stream
    docs = [{"docid": doc["docid"], "lang": lang, "title": doc["title"], "text": doc["text"]} for doc in sample]
    return docs


def clean_text(text: str) -> str:
    """Nettoie le texte en entrée."""
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('“', '"').replace('”', '"').replace('«', '"').replace('»', '"').replace("’", "'")
    text = text.replace('\n', '. ')
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    text = text.replace('\\n', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\!{2,}', '!', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&\w+;', '', text)
    text = re.sub(r'[^\x20-\x7EÀ-ÿ€£$¥•–—’“”…°²³µ·]', '', text)
    return text.strip()


def chunking(df: pd.DataFrame) -> pd.DataFrame:
    """Découpe le dataframe en chunks plus petits."""
    chunk_text, chunk_meta = [], []
    for _, row in df.iterrows():
        sentences = row["text"].split(". ")
        for sentence in sentences:
            if sentence:  # Ne pas ajouter de phrases vides
                chunk_text.append(sentence)
                chunk_meta.append({"title": row["title"], "docid": row["docid"], "lang": row["lang"]})
    df_chunk = pd.DataFrame({"text": chunk_text, "meta": chunk_meta})
    return df_chunk.drop_duplicates(subset=["text"]).dropna(subset=["text"])


def contains_explicit_1_january(text: str) -> bool:
    patterns = [r"\b0?1[\/\-\. ]?0?1\b", r"\b(1er|1|01)[^\d]?(janvier|january)\b",
                r"\b(janvier|january)[^\d]*(1er|1|01)\b", r"\b(1st|first) of (january|janvier)\b"]
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE): return True
    return False


def has_explicit_date_pattern(text: str) -> bool:
    numeric_patterns = [r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", r"\b\d{4}\b"]
    textual_patterns = [
        r"\b\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{2,4}\b"]
    for pattern in numeric_patterns + textual_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE): return True
    return False


def split_on_conjunctions(text: str) -> list:
    text = re.sub(r"\([^)]*\)", "", text)
    return re.split(r"\s+(et|and|,|;)\s+", text)


def is_pure_date_expression(text: str) -> bool:
    patterns = [r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b",
                r"\b\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b",
                r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b"]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def extract_date(df: pd.DataFrame) -> pd.DataFrame:
    parsers = [parser for parser in default_parsers if parser != 'relative-time']
    for i, row in df.iterrows():
        lang = row["meta"]["lang"]
        res_list = []
        try:
            segments = [seg.strip() for seg in split_on_conjunctions(row["text"]) if
                        seg.strip().lower() not in ["et", "and", ",", ";"] and seg.strip() != ""]
            for segment in segments:
                dates_found = search_dates(segment, languages=["fr", "en"],
                                           settings={'PREFER_DAY_OF_MONTH': 'first', 'PREFER_MONTH_OF_YEAR': 'first',
                                                     'DATE_ORDER': 'DMY' if lang == 'fr' else 'MDY',
                                                     'PARSERS': parsers})
                if dates_found:
                    for matched_text, result in dates_found:
                        res = {}
                        if result.day == 1 and result.month == 1:
                            if contains_explicit_1_january(matched_text):
                                res["year"] = result.year if has_explicit_date_pattern(matched_text) else None
                                res["month"] = result.month
                                res["day"] = result.day
                            else:
                                res["year"] = result.year if has_explicit_date_pattern(matched_text) else None
                                res["month"] = None
                                res["day"] = None
                        else:
                            res["year"] = result.year if has_explicit_date_pattern(matched_text) else None
                            res["month"] = result.month
                            res["day"] = result.day
                        res["hour"] = result.hour if result.hour else None
                        res["minute"] = result.minute if result.minute else None
                        res["second"] = result.second if result.second else None
                        if res["day"] and res["month"] and not is_pure_date_expression(matched_text): break
                        if res["day"] or res["month"] or res["year"]: res_list.append(res)
        except Exception as e:
            res_list = [{"error": str(e)}]
        df.at[i, "meta"]["dates"] = res_list
    return df


def post_treatment_bert_entities(entities: list[tuple[str, str]]) -> list[tuple[str, str]]:
    cleaned, current, label = [], "", None
    for word, tag in entities:
        if word.startswith("##"):
            current += word[2:]
        else:
            if current: cleaned.append((current.strip(), label))
            current, label = word, tag
    if current: cleaned.append((current.strip(), label))
    return [(w, t) for w, t in cleaned if len(w) >= 3 and (len(w) > 3 or t != "MISC")]


def enrich_df_with_ner_pipe(df_chunk: pd.DataFrame) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
    model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
    bert_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", batch_size=128,
                        device=0 if torch.cuda.is_available() else -1)
    if isinstance(df_chunk.iloc[0]["meta"], str): df_chunk["meta"] = df_chunk["meta"].apply(ast.literal_eval)
    for index, row in df_chunk.iterrows():
        text, final_ents = row["text"], []
        ents = [(ent['word'], ent['entity_group']) for ent in bert_ner(text)]
        ents = post_treatment_bert_entities(ents)
        for ent_text, ent_label in ents:
            if ent_label is not None and ent_label != "None" and ent_label != "DATE":
                final_ents.append((ent_text, ent_label))
        meta = df_chunk.at[index, "meta"]
        meta["entities"] = final_ents
        df_chunk.at[index, "meta"] = meta
    return df_chunk


# ==============================================================================
# SECTION 2: FONCTIONS POUR LE BENCHMARK DE GÉNÉRATION QA
# ==============================================================================

def qa_benchmark_worker_batch(chunks_subset, gpu_id, model_id):
    """
    Worker OPTIMISÉ qui traite les chunks par lots (batch) et gère différents prompts.
    """
    global qa_pipeline
    device = f'cuda:{gpu_id}'
    results = []
    total_time = 0

    try:
        qa_pipeline = pipeline("text2text-generation", model=model_id, tokenizer=model_id, device=device,
                               trust_remote_code=True)
        print(f"Worker sur GPU:{gpu_id}: Pipeline pour '{model_id}' chargé.")
    except Exception as e:
        print(f"Erreur critique lors du chargement du modèle {model_id} sur GPU:{gpu_id}: {e}")
        return [{"chunk": chunk, "generated_text": f"ERREUR CHARGEMENT: {e}"} for chunk in chunks_subset], 0

    start_time = time.time()
    try:
        # ### MODIFIÉ ### - Prompt adaptatif
        if "flan-t5" in model_id:
            # Prompt plus explicite pour les modèles instruction-tuned
            input_texts = [f"Please generate a question based on the following text: '{chunk}'" for chunk in
                           chunks_subset]
        else:
            # Prompt standard pour les modèles fine-tunés sur la génération de QA
            input_texts = [f"generate question: {chunk}" for chunk in chunks_subset]

        # ### MODIFIÉ ### - Taille de lot plus sûre
        batch_size = 16 if "large" in model_id else 64

        generated_outputs = qa_pipeline(input_texts, max_length=128, num_return_sequences=1, batch_size=batch_size)

        for i, output in enumerate(generated_outputs):
            # ### MODIFIÉ ### - On garde toute la sortie pour une évaluation robuste
            generated_text = output[0]['generated_text']
            results.append({"chunk": chunks_subset[i], "generated_text": generated_text})

    except Exception as e:
        print(f"Erreur de génération en batch sur GPU:{gpu_id}. Erreur: {e}")
        for chunk in chunks_subset:
            results.append({"chunk": chunk, "generated_text": f"ERREUR GENERATION: {e}"})

    total_time = time.time() - start_time

    del qa_pipeline
    torch.cuda.empty_cache()
    return results, total_time


# ==============================================================================
# SECTION 3: FONCTION PRINCIPALE ORCHESTRANT LE TOUT
# ==============================================================================

def main():
    # ... (Le code de prétraitement de la SECTION 1 est identique) ...
    checkpoint_file = "checkpoint__df_chunk_ready.parquet"
    if os.path.exists(checkpoint_file):
        print(f"[INFO] Chargement des données prétraitées depuis '{checkpoint_file}'...")
        df_chunk = pd.read_parquet(checkpoint_file)
    else:
        print("[INFO] Démarrage du pipeline de prétraitement.")
        languages, max_docs_per_lang = ['fr', 'en'], 10000
        results = Parallel(n_jobs=len(languages))(delayed(load_lang)(lang, max_docs_per_lang) for lang in languages)
        all_docs = [doc for lang_docs in results for doc in lang_docs]
        df = pd.DataFrame(all_docs)
        print("[INFO] Nettoyage des textes...")
        df["text"] = df["text"].progress_apply(clean_text)
        df_chunk = chunking(df)
        print(f"[INFO] Sauvegarde du DataFrame prétraité dans '{checkpoint_file}'.")
        df_chunk.to_parquet(checkpoint_file)

    print(f"\n[INFO] Prétraitement terminé. {len(df_chunk)} chunks prêts.")
    sample_size = 10000
    df_chunk_sample = df_chunk.sample(n=min(len(df_chunk), sample_size), random_state=42)
    all_chunks = df_chunk_sample['text'].dropna().tolist()
    print(f"Utilisation d'un échantillon de {len(all_chunks)} chunks pour le benchmark.")

    # --- BENCHMARK DES MODÈLES QA ---
    if not torch.cuda.is_available(): print("Aucun GPU CUDA n'est disponible."); return
    num_gpus = torch.cuda.device_count()
    print(f"{num_gpus} GPU(s) détecté(s).")

    bertscore = evaluate.load("bertscore")
    rouge = evaluate.load("rouge")
    benchmark_results = []

    for model_info in tqdm(MODELS_TO_BENCHMARK, desc="Benchmarking Models"):
        model_id = model_info["model_id"]
        tqdm.write(f"\n{'=' * 80}\nBenchmarking du modèle QA : {model_id}\n{'=' * 80}")

        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True)
            model_params = model.num_parameters() / 1_000_000
            del model
            torch.cuda.empty_cache()
        except Exception:
            model_params = -1

        chunks_per_gpu = np.array_split(all_chunks, num_gpus)
        pool_args = [(chunks.tolist(), i, model_id) for i, chunks in enumerate(chunks_per_gpu) if len(chunks) > 0]

        total_gen_time, all_generated_data = 0, []
        with mp.get_context('spawn').Pool(processes=num_gpus) as pool:
            worker_outputs = pool.starmap(qa_benchmark_worker_batch, pool_args)
            for data_list, time_taken in worker_outputs:
                all_generated_data.extend(data_list)
                total_gen_time += time_taken

        if not all_generated_data: tqdm.write(f"Aucun résultat généré pour {model_id}."); continue

        generated_df = pd.DataFrame(all_generated_data)

        # ### MODIFIÉ ### - Évaluation robuste
        # On compare le chunk original à TOUTE la sortie générée.
        predictions = generated_df["generated_text"].tolist()
        references = generated_df["chunk"].tolist()

        tqdm.write("Calcul des métriques de qualité...")
        try:
            bert_results = bertscore.compute(predictions=predictions, references=references, lang="fr", device='cuda:0',
                                             batch_size=16)
            avg_bert_f1 = np.mean(bert_results['f1']) if bert_results['f1'] else -1.0
        except Exception as e:
            tqdm.write(f"Erreur BERTScore: {e}")
            avg_bert_f1 = -1

        try:
            rouge_results = rouge.compute(predictions=predictions, references=references)
            rouge_l_score = rouge_results['rougeL']
        except Exception as e:
            tqdm.write(f"Erreur ROUGE: {e}")
            rouge_l_score = -1

        results = {
            "model_id": model_id,
            "size_category": model_info["size_category"],
            "model_complexity_M_params": round(model_params, 2),
            "generation_time_seconds": round(total_gen_time, 2),
            "chunks_per_second": round(len(all_chunks) / total_gen_time, 2) if total_gen_time > 0 else 0,
            "bertscore_f1": round(avg_bert_f1, 4),
            "rougeL": round(rouge_l_score, 4),
        }
        benchmark_results.append(results)
        tqdm.write(f"Résultats pour {model_id}: {results}")

    final_df = pd.DataFrame(benchmark_results)
    output_path = "qa_generation_benchmark_results.csv"
    final_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 80}\nBenchmark QA terminé !\n{'=' * 80}")
    print("Tableau récapitulatif des résultats :")
    print(final_df.to_string())
    print(f"\nLes résultats complets sont sauvegardés dans : {output_path}")


if __name__ == "__main__":
    main()
