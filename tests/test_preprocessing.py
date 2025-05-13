from preprocessing import preprocess_data
import pandas as pd

languages = ['fr', 'en']
max_docs_per_lang = 10 # taille du dataset (x2 car français + anglais)

res = preprocess_data(languages, max_docs_per_lang, 1)

print(res)
print(res.head())
print(res.info())

print(res["meta"].head())

print(res["meta"].info())

res.to_csv("test.csv", index=False)