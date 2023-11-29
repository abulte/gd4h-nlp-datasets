# gd4h-nlp-datasets

## Introduction

Experiments with GD4H semantics and data.gouv.fr's datasets catalog.

Goal: find datasets on data.gouv.fr's related to various GD4H-specific keywords / semantic space.

Different methods have been explored, and are explained below.

### Input

Thoses scripts are based on:
- [data.gouv.fr's datasets CSV export](https://www.data.gouv.fr/fr/datasets/catalogue-des-donnees-de-data-gouv-fr/), up to date at the time of computation
- `tags.txt` as semantic field for the legacy word2vec output of august 2023
- `keywords.yml` as thematic semantic field(s) used for the other outputs (november 2023)

### Output

Computed results are stored in the `output-filtered` directory. Each computation is under a timestamped directory, with the following conventionnal filenames:
- `{method}-output[_{theme}]-mapped.csv` for the top 5000 (if applicable) datasets detected with `method`, with a `in_gd4h` column that can be True or False depending on if the dataset is linked to a GD4H org
- `{method}-output[_{theme}]-mapped-filtered.csv` for a filtered version of the above file with `in_gd4h == True`

Note the `_{theme}` optionnal component: when applicable, it identifies a given theme defined in `keywords.yml` (top level element). `all` is a special theme that concatenates all the keywords for all the themes.

## word2vec

Lives in `learn_word2vec.py`.

This method trains a [custom Word2Vec model with gensim](https://radimrehurek.com/gensim/models/word2vec.html) (cbog), using a corpus generated from the data.gouv.fr's datasets catalog. The corpus is tokenized beforehand using NLTK french tokenizer and some custom adjustements (remove markdown, min token size, remove contractions). Corpus contains title and description of every dataset in the catalog, except archived ones.

It then computes :
- a vector for a list of keyword expressions (see input), expressed in the discrete vector space of the trained model
- a similiar vector for every dataset in the corpus

Finally, dataset and keywords vectors are compared via cosine similiarity and given a score. Datasets with the highest score are deeemed semantically closer to the given keywords.

Outputs for this method:
- `20230825` with legacy `tags.txt` as input
- `20231123-175957` with `keywords.yml` as input, computing a match list for every theme — some themes could not been expressed as-is in the vector space and thus are not computed
- `20231123-191744-tokenize-keywords` with the same method as above, but by tokenizing keywords before computing similarities

**20231123-175957 seems to yield the best results, after a quick manual review.**

## llm

⚠ `llm` refers to [the toolkit used](https://llm.datasette.io), no Large Language Model has been ~~harmed~~ used during the experiments.

Lives in `learn_llm.py`.

This methods uses two pretrained models to compute vector embeddings for data.gouv.fr's catalog:
- `all-MiniLM-L6-v2`
- `distiluse-base-multilingual-cased-v1`

The approach is similiar to word2vec: compute cosine similarity between keywords vectors and datasets vectors, assign a score.

The idea here vs word2vec was to use a more integrated approach: use pretrained models and tools to handle the vectors. It does not seem to yield better results though.

Outputs for this method:
- `20231124-152650` with `all-MiniLM-L6-v2`
- `20231124-152742` with `distiluse-base-multilingual-cased-v1`

## top2vec

Live in `learn_top2vec.py` for training `top2vec.ipynb` and `top2vec_search.ipynb` for exploration.

This uses the [Top2Vec](https://github.com/ddangelov/Top2Vec) to explore a different approach: topic modeling.

The embedding model used is `distiluse-base-multilingual-cased`.

The general idea is to let the library create some clusters (topics) and try to select some of those topics based on our inputs.

The corpus contains the title, description and tags (untouched before being fed to the model) of every dataset in the catalog, except archived ones.

Once the topics have been infered, two methods have been applied:
- [search_topics](https://top2vec.readthedocs.io/en/latest/api.html#top2vec.Top2Vec.Top2Vec.search_topics): semantic search of topics using keywords matching. The keywords are those defined for each themes in `keywords.yml`.
- [query_topics](https://top2vec.readthedocs.io/en/latest/api.html#top2vec.Top2Vec.Top2Vec.query_topics): semantic search of topics using text query. The text queries are the name of the themes defined in `keywords.yml` (feeding the associated keywords did not seem to change the results). In this method we're merging multiple topics and we're using the search score for the topic to normalize the datasets' score in each topic.

Similarly to other methods, the datasets associated to the selected topics have been extracted in a CSV file and scored.

This method looks promising because exploring the topics manually shows some very meaningful clusters. See the notebook for example. Unfortunately it's difficult to select those topics automatically — some very noisy and irrelevant ones are always popping up — and the number of datasets in a topic can be quite small.

Outputs for this method:
- `20231127-125722` with `search_topics`
- `20231127-170325` with `query_topics`
- `20231129-115554` with modified `search_topics` and a new model
