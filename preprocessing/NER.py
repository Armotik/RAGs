import pandas as pd
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

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
        )
    return _nlp_models[lang]

def post_treatment_bert_entities(entities: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Post-process the entities detected by BERT to clean them up before sending them to spaCy.
    :param entities: the list of entities detected by BERT
    :return: the cleaned list of entities
    """
    cleaned = []
    current = ""
    label = None

    for word, tag in entities:
        if word.startswith("##"):
            current += word[2:]
        else:
            if current:
                cleaned.append((current.strip(), label))
            current = word
            label = tag

    if current:
        cleaned.append((current.strip(), label))

    cleaned = [(w, t) for w, t in cleaned if len(w) > 2 and w.lower() not in {"l'", "’", "le", "la"}]

    return cleaned

def enrich_df_with_ner_pipe(df_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the DataFrame with Named Entity Recognition (NER) using BERT and spaCy.
    :param df_chunk: the DataFrame to enrich
    :return: the enriched DataFrame
    """
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
    bert_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    entities_all = []

    for index, row in df_chunk.iterrows():

        text = row["text"]
        lang = row["meta"]["lang"]

        # BERT NER
        ents = [(ent['word'], ent['entity_group']) for ent in bert_ner(text)]
        ents = post_treatment_bert_entities(ents)

        final_ents = []
        for ent_text, ent_label in ents:
            if ent_label == "None" or ent_label == "DATE": # skip None labels and DATE (Date Extraction is handled separately)
                continue
            if ent_label == "MISC":
                spacy_nlp = get_nlp(lang)
                doc = spacy_nlp(ent_text)
                sub_ents = [(e.text, e.label_) for e in doc.ents]
                if sub_ents:
                    final_ents.extend(sub_ents)
            else:
                final_ents.append((ent_text, ent_label))

        entities_all.append(final_ents)

    df_chunk = df_chunk.copy()
    df_chunk["entities"] = entities_all
    return df_chunk

