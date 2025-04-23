import pandas as pd
import spacy

_nlp_models = {}

def get_nlp(lang : str) -> spacy.Language:
    """
    Load the spaCy model for the given language if not already loaded.
    :param lang: the language to load the model for
    :return: the spaCy model for the given language
    """
    if lang not in _nlp_models:
        _nlp_models[lang] = spacy.load(
            "fr_core_news_sm" if lang == "fr" else "en_core_web_sm",
            disable=["tagger", "parser"]
        )
    return _nlp_models[lang]

def enrich_df_with_ner_pipe(df_chunk : pd.DataFrame) -> pd.DataFrame: # À AMÉLIORER
    """
    Enrich the dataframe with named entity recognition (NER) using spaCy.
    :param df_chunk: the dataframe chunk to enrich
    :return: the enriched dataframe chunk
    """
    enriched_rows = []

    for lang in ["fr", "en"]:
        sub_df = df_chunk[df_chunk["meta"].apply(lambda m: m.get("lang") == lang)]
        if sub_df.empty:
            continue

        nlp = get_nlp(lang)
        texts = sub_df["text"].tolist()
        metas = sub_df["meta"].tolist()

        docs = list(nlp.pipe(texts, batch_size=256))

        for meta, doc in zip(metas, docs):
            entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
            meta = meta.copy()
            meta["entities"] = entities
            enriched_rows.append({"text": doc.text, "meta": meta})

    return pd.DataFrame(enriched_rows)