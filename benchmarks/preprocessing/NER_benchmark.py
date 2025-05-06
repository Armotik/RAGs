from datasets import load_dataset
from itertools import islice
from joblib import Parallel, delayed
import unicodedata
import re
import pandas as pd
from dateparser.search import search_dates
from dateparser_data.settings import default_parsers
import numpy as np
import spacy
from time import time
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import torch

def load_lang(lang: str, max_docs: int) -> list:
    """
    Load the dataset for a specific language and return a list of documents.
    :param lang: language code (e.g., 'fr' or 'en')
    :param max_docs: maximum number of documents to a load
    :return: list of documents
    """
    print(f"[{lang.upper()}] Chargement...")
    dataset_stream = load_dataset('miracl/miracl-corpus', lang, split='train', streaming=True, trust_remote_code=True)
    sample = islice(dataset_stream, max_docs) if max_docs > 0 else dataset_stream

    docs = []
    for doc in sample:
        docs.append({
            "docid": doc["docid"],
            "lang": lang,
            "title": doc["title"],
            "text": doc["text"]
        })
    return docs

languages = ['fr', 'en']
max_docs_per_lang = 10  # taille du dataset (x2 car français + anglais)

results = Parallel(n_jobs=len(languages))(
    delayed(load_lang)(lang, max_docs_per_lang) for lang in languages
)

print(f"[INFO] {len(results)} langues chargées avec succès.")

all_docs = [doc for lang_docs in results for doc in lang_docs]

def clean_text(text: str) -> str:
    """
    Clean the input text by performing various preprocessing steps.
    :param text: the text to clean
    :return: cleaned text
    """

    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)

    # Replacing typographical quotation marks with single quotation marks
    text = text.replace('“', '"').replace('”', '"').replace('«', '"').replace('»', '"').replace("’", "'")

    # Replace \n with a space (or a period + space if it breaks a sentence)
    text = text.replace('\n', '. ')

    # Delete invisible control characters (except those already managed)
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

    # Delete multiple spaces
    text = text.replace('\\n', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)

    # Delete multiple spaces around punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\!{2,}', '!', text)

    # Delete spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)

    # Delete HTML tags and HTML entities
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&\w+;', '', text)

    # Delete unprintable characters or orphan symbols
    text = re.sub(r'[^\x20-\x7EÀ-ÿ€£$¥•–—’“”…°²³µ·]', '', text)

    return text.strip()

print("[INFO] Nettoyage des textes...")

df = pd.DataFrame(all_docs)
df["title"] = df["title"].apply(clean_text)
df["text"] = df["text"].apply(clean_text)

print("[INFO] Nettoyage terminé.")

def chunking(df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    """
    Chunking the dataframe into smaller chunks of a specified size.
    :param df: the dataframe to chunk
    :param chunk_size: the size of each chunk
    :return: a new dataframe with the chunks
    """

    # TODO : chunking by chunk_size

    chunk_text = []
    chunk_meta = []

    for index, row in df.iterrows():
        text = row["text"]
        title = row["title"]
        docid = row["docid"]
        lang = row["lang"]
        sentences = text.split(". ")

        for sentence in sentences:
            chunk_text.append(sentence)
            chunk_meta.append({
                "title": title,
                "docid": docid,
                "lang": lang
            })

    if len(chunk_text) != len(chunk_meta):
        raise ValueError("Chunk text and metadata lengths do not match.")

    df_chunk = pd.DataFrame({"text": chunk_text, "meta": chunk_meta})
    df_chunk = df_chunk.drop_duplicates(subset=["text"])
    df_chunk = df_chunk.dropna(subset=["text"])

    return df_chunk

print("[INFO] Découpage des textes en phrases...")

df_chunk = chunking(df, 0)

print("[INFO] Découpage terminé.")

def contains_explicit_1_january(text: str) -> bool:
    """
    Check if the text contains an explicit mention of 1st January.
    :param text: the text to check
    :return: True if the text contains an explicit mention of 1st January, False otherwise
    """
    patterns = [
        r"\b0?1[\/\-\. ]?0?1\b",  # 01/01, 1/1, 01-01, etc.
        r"\b(1er|1|01)[^\d]?(janvier|january)\b",  # 1 janvier, 1er janvier, 01 janvier
        r"\b(janvier|january)[^\d]*(1er|1|01)\b",  # janvier 1, january 1st
        r"\b(1st|first) of (january|janvier)\b"  # 1st of January
    ]
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def has_explicit_date_pattern(text: str) -> bool:
    """
    Check if the text contains an explicit date pattern. In case of ambiguous date.
    :param text: the text to check
    :return: True if the text contains an explicit date pattern, False otherwise
    """
    numeric_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}\b"
    ]

    textual_patterns = [
        r"\b\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|"
        r"january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{2,4}\b"
    ]

    all_patterns = numeric_patterns + textual_patterns

    for pattern in all_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def split_on_conjunctions(text: str) -> list:
    """
    Split the text on conjunctions (et, and, , or ;).
    :param text: the text to split
    :return: the list of segments
    """

    text = re.sub(r"\([^)]*\)", "", text)

    return re.split(r"\s+(et|and|,|;)\s+", text)


def is_pure_date_expression(text: str) -> bool:
    """
    Check if the text is a pure date expression.
    :param text: the text to check
    :return: True if the text is a pure date expression, False otherwise
    """
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b",  # 15/04[/2025]
        r"\b\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b",
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b"
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def extract_date(df: pd.DataFrame) -> pd.DataFrame:
    parsers = [parser for parser in default_parsers if parser != 'relative-time']

    for i, row in df.iterrows():
        lang = row["meta"]["lang"]
        res_list = []

        try:
            # On divise le texte sur les conjonctions
            segments = [seg.strip() for seg in split_on_conjunctions(row["text"]) if
                        seg.strip().lower() not in ["et", "and", ",", ";"] and seg.strip() != ""]

            for segment in segments:
                dates_found = search_dates(
                    segment,
                    languages=["fr", "en"],
                    settings={
                        'PREFER_DAY_OF_MONTH': 'first',
                        'PREFER_MONTH_OF_YEAR': 'first',
                        'DATE_ORDER': 'DMY' if lang == 'fr' else 'MDY',
                        'PARSERS': parsers,
                    }
                )

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

                        # False positive check
                        if res["day"] and res["month"] and not is_pure_date_expression(matched_text):
                            break

                        if res["day"] or res["month"] or res["year"]:
                            res_list.append(res)

        except Exception as e:
            res_list = [{"error": str(e)}]

        df.at[i, "meta"]["dates"] = res_list

    return df

print("[INFO] Extraction des dates...")

df_chunk = extract_date(df_chunk)

print("[INFO] Extraction des dates terminée.")

chunks = np.array_split(df_chunk, 256)

models = {
    "spacy-fr-core-news-md": ["fr", "fr_core_news_sm", "spacy"],
    "spacy-fr_dep_news_trf": ["fr", "fr_dep_news_trf", "spacy"],
    "spacy-en-core-web-md": ["en", "en_core_web_md", "spacy"],
    "spacy-en_core_web_trf": ["en", "en_core_web_trf", "spacy"],
    "spacy-xx_ent_wiki_sm": ["multilingual", "xx_ent_wiki_sm", "spacy-first"],
    "spacy-xx_sent_ud_sm": ["multilingual", "xx_sent_ud_sm", "spacy-first"],
    "wikineural-multilingual-ner": ["multilingual", "Babelscape/wikineural-multilingual-ner", "transformers"],
    "bert-base-multilingual-cased": ["multilingual", "google-bert/bert-base-multilingual-cased", "transformers"],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] Device utilisé :", device)

def compute_tfidf(entities):
    if not entities or all(e.strip() == "" for e in entities):
        return 0.0
    try:
        tfidf = TfidfVectorizer().fit_transform([" ".join(entities)])
        return float(np.mean(tfidf.sum(axis=1)))
    except ValueError:
        return 0.0

def compute_seq_score(entity_list):
    lengths = [len(ent.split()) for ent in entity_list]
    return {
        "avg_len": np.mean(lengths) if lengths else 0,
        "uniq_ratio": len(set(entity_list)) / len(entity_list) if entity_list else 0
    }

def compute_avg_entity_length(entities):
    if not entities:
        return 0.0
    return float(np.mean([len(e) for e in entities if isinstance(e, str)]))

def compute_entity_stability(entities_base, entities_secondary):
    if not entities_base or not entities_secondary:
        return 0.0
    set_base = set(entities_base)
    set_secondary = set(entities_secondary)
    intersection = set_base.intersection(set_secondary)
    union = set_base.union(set_secondary)
    return len(intersection) / len(union) if union else 0.0


def run_transformers(model_name, text):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    pipe = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
    results = pipe(text)
    entities = [r["word"] for r in results]
    return entities

def run_spacy(model_name, text):
    nlp = spacy.load(model_name)
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

def benchmark_ner(df, max_rows=20):
    results = []
    df = df.head(max_rows)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Benchmarking"):
        text = row["text"]
        meta = row["meta"]
        lang = meta["lang"]

        for name, (model_lang, model_id, framework) in models.items():
            if framework == "spacy":
                continue  # spaCy uniquement utilisé en phase 2

            # === Phase 1 : Modèle principal ===
            start = time()
            entities = run_transformers(model_id, text) if framework == "transformers" else run_spacy(model_id, text)
            duration = time() - start
            tfidf = compute_tfidf(entities)
            seq = compute_seq_score(entities)

            results.append({
                "text": text,
                "text_lang": lang,
                "model": name,
                "phase": "base",
                "duration": duration,
                "entity_count": len(entities),
                "tfidf_score": tfidf,
                "seqscore": seq,
                "avg_entity_length": compute_avg_entity_length(entities),
                "entity_stability": compute_entity_stability(entities, entities),
                "entities": entities
            })

            # === Phase 2 : Second passage avec spaCy
            for ent in entities:

                for spacy_name, (s_lang, s_model_id, s_framework) in models.items():
                    if s_framework == "spacy" and s_lang == lang:

                        start2 = time()

                        ents2 = run_spacy(s_model_id, ent)

                        d2 = time() - start2
                        d2 = duration + d2

                        tfidf2 = compute_tfidf(ents2)
                        seq2 = compute_seq_score(ents2)
                        results.append({
                            "text": ent,
                            "text_lang": lang,
                            "model": f"{name} → {spacy_name}",
                            "phase": "refinement_with_spacy",
                            "duration": d2,
                            "entity_count": len(ents2),
                            "tfidf_score": tfidf2,
                            "seqscore": seq2,
                            "avg_entity_length": compute_avg_entity_length(ents2),
                            "entity_stability": compute_entity_stability(entities, ents2),
                            "entities": ents2
                        })
    return pd.DataFrame(results)

max_rows = len(df_chunk) if device == "cuda" else 20

print("[INFO] Lancement du benchmark NER...")

results_df = benchmark_ner(df_chunk, max_rows=10)
results_df.to_csv("./NER_benchmark_results.csv", index=False)

print("[INFO] Benchmark terminé.")