# Retrieval Augmented Generattion for Historical Newspaper

## Stage Licence 3 Informatique - La Rochelle Université

### Structure : 
- Laboratoire Informatique Image Intéractions - La Rochelle Université
- Équipe Images et Contenus (IC)

### Encadrant :
 - Souhail Bakkali

---

## Objectifs du stage 

L’objectif principal du stage est de renforcer le pipeline RAG afin d'améliorer la récupération 
d’articles, la génération de résumés et la réponse aux questions à partir d’un corpus de 
journaux historiques. Dans une seconde phase, si le temps le permet, une extension vers un 
RAG multimodal (vision + langage) pourra être explorée

Les objectifs détaillés sont les suivants :
1. **Améliorer le pipeline RAG existant**
    -  Optimisation du module de récupération d’informations en exploitant des 
moteurs sémantiques (FAISS, Elasticsearch).
      - Intégration de techniques avancées pour atténuer l’impact des erreurs d’OCR 
(correction automatique, re-ranking). 
      - Affinement du modèle de génération (GPT, T5, LLaMA) pour améliorer la 
qualité des réponses et des résumés générés. 
      - Définition et mise en place de métriques d’évaluation spécifiques permettant 
d’estimer la pertinence des générations sans nécessiter de vérité terrain (ground 
truth).
2. Explorer une approche multimodale (si le temps le permet)
   - Intégration de modèles vision-langage (ex. BLIP, LayoutLM) pour enrichir le 
   processus de récupération et de génération d’informations en tenant compte des 
   éléments visuels des journaux (titres, colonnes, images, mise en page).
   - Étude de l’impact des représentations visuelles sur la qualité de l’agrégation et 
   de la structuration des articles.

## Méthodologie :

Le stage se déroulera en plusieurs étapes :
- Analyse de l’existant : prise en main du pipeline RAG actuel, compréhension des 
limitations et définition des axes d’amélioration.
- Optimisation de la récupération sémantique : amélioration du module de recherche 
avec des techniques avancées (re-ranking, correction d’OCR, intégration d’entités 
nommées).
- Optimisation du modèle génératif : expérimentation avec différents modèles de 
langage et ajustement des paramètres pour une meilleure génération de résumés et 
réponses aux questions.
- Définition des métriques d’évaluation : mise en place d’indicateurs pour mesurer la 
qualité des résultats (ex. pertinence des réponses, cohérence des résumés).
- Exploration multimodale (optionnelle) : intégration de l’information visuelle dans le 
pipeline pour enrichir la compréhension et l’agrégation des articles.

## Références bibliographiques : 

- The Trung Tran, Carlos-Emiliano González-Gallardo, Antoine Doucet. Retrieval Augmented Generation
for Historical Newspapers. ACM/IEEE-CS Joint Conference on Digital Libraries (JCDL), Dec
2024, Hong Kong, China. ￿ hal-0479608
- Shangyu Wu, Ying Xiong*, Yufei Cui, Haolun Wu, Can Chen, Ye Yuan,
Lianming Huang, Xue Liu, Tei-Wei Kuo, Nan Guan, and Chun Jason Xue.
Retrieval-Augmented Generation for Natural Language Processing: A Survey.

---

Anthony Mudet

La Rochelle Université

2025