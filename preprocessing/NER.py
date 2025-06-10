import pandas as pd
import torch.cuda
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import ast

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

    cleaned = [(w, t) for w, t in cleaned if len(w) >= 3 and (len(w) > 3 or t != "MISC")]
    return cleaned

def enrich_df_with_ner_pipe(df_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the DataFrame with Named Entity Recognition (NER) using BERT and spaCy, storing results into meta["entities"].
    :param df_chunk: the DataFrame to enrich
    :return: the enriched DataFrame
    """
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
    model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
    bert_ner = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        batch_size=128
    )

    if isinstance(df_chunk.iloc[0]["meta"], str):
        df_chunk["meta"] = df_chunk["meta"].apply(lambda x: ast.literal_eval(x))

    for index, row in df_chunk.iterrows():
        text = row["text"]

        ents = [(ent['word'], ent['entity_group']) for ent in bert_ner(text)]
        ents = post_treatment_bert_entities(ents)

        final_ents = []
        for ent_text, ent_label in ents:
            if ent_label is None or ent_label == "None" or ent_label == "DATE":
                continue

            final_ents.append((ent_text, ent_label))

        meta = df_chunk.at[index, "meta"]
        meta["entities"] = final_ents
        df_chunk.at[index, "meta"] = meta

    torch.cuda.empty_cache()
    del bert_ner
    del tokenizer
    del model

    return df_chunk

