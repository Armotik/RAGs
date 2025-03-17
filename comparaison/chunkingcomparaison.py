from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import NLTKTextSplitter
from time import time
import nltk

from tools.pdfloader import PDFLoader

class ChunkingComparaison:
    def __init__(self, path):
        self.path = path
        self.pages = []

    def load(self):
        loader = PDFLoader(self.path)
        for page in loader.load():
            self.pages.append(page)

        return self.pages

    def compare_time(self):
        spacy_splitter = SpacyTextSplitter()
        nltk_splitter = NLTKTextSplitter()

        spacy_time = 0
        nltk_time = 0

        for page in self.pages:
            start = time()
            spacy_splitter.split_text(page.page_content)
            spacy_time += time() - start

            start = time()
            nltk_splitter.split_text(page.page_content)
            nltk_time += time() - start

        return spacy_time, nltk_time

    def compare_chunks(self):
        spacy_splitter = SpacyTextSplitter()
        nltk_splitter = NLTKTextSplitter()

        spacy_chunks = []
        nltk_chunks = []

        for page in self.pages:
            spacy_chunks.append(spacy_splitter.split_text(page.page_content))
            nltk_chunks.append(nltk_splitter.split_text(page.page_content))

        return spacy_chunks, nltk_chunks


if __name__ == "__main__":
    nltk.download('punkt_tab')
    path = "../data/article.pdf"
    chunking = ChunkingComparaison(path)
    chunking.load()
    spacy_time, nltk_time = chunking.compare_time()
    spacy_chunks, nltk_chunks = chunking.compare_chunks()

    print("Comparaison du temps de traitement de Spacy et NLTK pour la chunking")
    print(f"Spacy time: {spacy_time}")
    print(f"NLTK time: {nltk_time}")
    print("-------------------")
    print("Comparaison des résultats de Spacy et NLTK pour les chunks")
    print(f"Spacy chunks: {spacy_chunks}")
    print(f"NLTK chunks: {nltk_chunks}")

    print("Taille des chunks")
    print(f"Spacy chunks: {len(spacy_chunks)}")
    print(f"NLTK chunks: {len(nltk_chunks)}")