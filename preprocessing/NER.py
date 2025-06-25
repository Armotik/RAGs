# NER.py
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import ast
import gc


class NERProcessor:
    """
    Une classe optimisée pour effectuer la reconnaissance d'entités nommées (NER) sur GPU.
    Elle charge le modèle une seule fois et traite les données en lots pour une performance maximale.
    """

    def __init__(self, model_name: str = "Babelscape/wikineural-multilingual-ner", device: str = "cuda:0"):
        """
        Initialise le processeur NER en chargeant le modèle et le tokenizer.
        """
        print(f"[NERProcessor] Initialisation sur le device : {device}")
        if not torch.cuda.is_available():
            raise RuntimeError("Le calcul sur GPU est demandé, mais CUDA n'est pas disponible.")

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)

        # NOTE: On retire torch.compile() car il est incompatible avec la pipeline NER de Hugging Face
        # et causait les erreurs "OptimizedModule" et "OutOfResources".

        self.bert_ner = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=self.device,
            batch_size=256  # Taille de lot agressive pour les gros GPUs
        )
        print("[NERProcessor] Pipeline NER prête.")

    def _post_process_entities(self, entities: list) -> list[tuple[str, str]]:
        """ Nettoie les entités brutes retournées par la pipeline. """
        cleaned = []
        for entity in entities:
            word = entity['word']
            tag = entity['entity_group']
            if word and len(word) >= 3 and tag != "MISC" and tag is not None and tag != "None":
                cleaned.append((word.strip(), tag))
        return cleaned

    def process_dataframe(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        """ Enrichit un DataFrame entier en traitant tous les textes en un seul lot. """
        if df_chunk.empty or 'text' not in df_chunk.columns:
            return df_chunk

        print("[NERProcessor] Extraction des textes pour le traitement par lot...")
        texts = df_chunk['text'].fillna('').tolist()

        print(f"[NERProcessor] Traitement de {len(texts)} textes par la pipeline NER...")
        all_ents_batches = self.bert_ner(texts)
        print("[NERProcessor] Traitement par lot terminé.")

        if 'meta' not in df_chunk.columns:
            df_chunk['meta'] = [{} for _ in range(len(df_chunk))]
        else:
            df_chunk['meta'] = df_chunk['meta'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, dict) else {}))

        final_entities_list = [self._post_process_entities(ents) for ents in all_ents_batches]

        df_chunk['entities'] = final_entities_list
        df_chunk['meta'] = df_chunk.apply(lambda row: {**row['meta'], 'entities': row['entities']}, axis=1)
        df_chunk = df_chunk.drop(columns=['entities'])

        return df_chunk

    def release(self):
        """Libère la mémoire GPU."""
        print("[NERProcessor] Libération des ressources NER...")
        del self.bert_ner, self.model, self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def enrich_df_with_ner_pipe(df_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction wrapper pour instancier, utiliser et détruire le processeur NER.
    """
    if df_chunk.empty:
        return df_chunk
    processor = None
    try:
        processor = NERProcessor()
        return processor.process_dataframe(df_chunk)
    finally:
        if processor:
            processor.release()
