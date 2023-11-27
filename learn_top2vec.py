from pathlib import Path

from minicli import cli, run
from top2vec import Top2Vec

from utils import load_raw_corpus


catalog = Path("datasets-filtered.csv")


@cli
def train_model(embedding_model="distiluse-base-multilingual-cased"):
    slugs, corpus = zip(*load_raw_corpus(catalog, delimiter=","))
    model = Top2Vec(list(corpus), embedding_model=embedding_model, document_ids=list(slugs))
    model.save(f"top2vec_{embedding_model}.bin")


if __name__ == "__main__":
    run()
