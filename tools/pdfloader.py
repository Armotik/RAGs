from langchain_community.document_loaders import PyPDFLoader

class PDFLoader:
    def __init__(self, path):
        self.path = path
        self.pages = []

    def load(self):
        loader = PyPDFLoader(self.path)
        for page in loader.lazy_load():
            self.pages.append(page)

        return self.pages