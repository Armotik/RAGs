from preprocessing import preprocess_data

import pandas as pd

###
###
###

languages = ['fr', 'en']
max_docs_per_lang = 100 # taille du dataset (x2 car français + anglais)

res = preprocess_data(languages, max_docs_per_lang, 64)

print(res)