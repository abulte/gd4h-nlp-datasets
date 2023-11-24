import csv

from datetime import datetime
from pathlib import Path

import numpy as np

from gensim.models import Word2Vec
from minicli import cli, run
from progressist import ProgressBar
from slugify import slugify
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_tokenized_corpus, load_raw_keywords, load_kw_file


@cli
def compute(catalog: Path, theme: str = "all", refresh=False, storage=None):
    if not storage:
        storage = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Vos documents prétraités (une liste de listes de mots)
    slugs, corpus = zip(*load_tokenized_corpus(catalog, refresh=refresh))

    # Création du modèle Word2Vec
    mpath = Path("word2vec-model.bin")
    if not mpath.exists() or refresh:
        print("Training model...")
        word2vec_model = Word2Vec(sentences=corpus, vector_size=100,
                                  window=5, min_count=1, workers=4)
        word2vec_model.save(mpath.name)
    else:
        print("Loading existing model...")
        word2vec_model = Word2Vec.load(mpath.name)

    # TODO: tokenize?
    keyword_expressions = load_raw_keywords(theme)

    expression_vectors = []
    for expression in keyword_expressions:
        expression_vector = np.mean(
            [word2vec_model.wv[word] for word in expression.split() if word in word2vec_model.wv],
            axis=0,
        )
        # TODO: measure this
        # sometimes, the expression can not be expressed in model vector space
        if expression_vector.shape:
            expression_vectors.append(expression_vector)

    if not expression_vectors:
        print("[ERROR] no vector could be computed for keyword corpus")
        return

    # compute similarities for each doc with expressions
    print("Computing similarities...")
    document_similarities = []
    bar = ProgressBar(total=len(corpus))
    for doc in bar.iter(corpus):
        # mean of word similarities with corpus for each word of document
        # XXX does this makes senses on own-trained model?
        doc_vector = np.mean(
            [word2vec_model.wv[word] for word in doc if word in word2vec_model.wv],
            axis=0,
        )
        similarities = cosine_similarity(expression_vectors, [doc_vector])
        # compute mean of similarities between doc and expressions
        mean_similarities = np.mean(similarities)
        document_similarities.append(mean_similarities)

    documents_with_similarity = list(zip(slugs, document_similarities))
    most_similar_documents = sorted(documents_with_similarity, key=lambda x: x[1], reverse=True)

    output_path = Path(f"./output/{storage}")
    output_file = f"word2vec-output_{slugify(theme)}.csv"
    output_path.mkdir(exist_ok=True)
    with (output_path / output_file).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["slug", "score"])
        writer.writeheader()
        for doc in most_similar_documents:
            writer.writerow({"slug": doc[0], "score": doc[1]})


@cli
def compute_all(catalog: Path, refresh: bool = False):
    themes = list(load_kw_file().keys())
    themes.append("all")
    storage = datetime.now().strftime('%Y%m%d-%H%M%S')
    for theme in themes:
        print(f'Computing similarities for theme "{theme}"')
        compute(catalog, theme=theme, refresh=refresh, storage=storage)


if __name__ == "__main__":
    run()
