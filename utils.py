import csv
import pickle
import re
import yaml

from pathlib import Path

import nltk

from markdown_it import MarkdownIt
from mdit_plain.renderer import RendererPlain
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from progressist import ProgressBar


def load_raw_corpus(source: Path, max: int | None = None) -> list:
    """
    Returns [("{slug}", "{text}")]
    """
    cur = 0
    corpus = []
    with source.open() as f:
        reader = csv.DictReader(f, delimiter=";")
        for line in reader:
            if max and cur == max:
                break

            if not line["archived"] == "False":
                continue

            cur += 1

            corpus.append((
                line["slug"],
                f'{line["title"]} {line["description"]}'
            ))

    return corpus


def load_kw_file() -> dict:
    with open("keywords.yml") as f:
        return yaml.safe_load(f)


def load_raw_keywords(theme) -> list:
    data = load_kw_file()
    if theme == "all":
        return list(set([tag for _, tags in data.items() for tag in tags]))
    else:
        return data[theme]


def tokenize(data: str, min_length: int = 2) -> list:
    """
    Tokenize words through nltk:
    - remove markdown formatting
    - remove stopword, short words and apostrophes
    """
    nltk.download("punkt")
    parser = MarkdownIt(renderer_cls=RendererPlain)
    french_stop_words = stopwords.words("french")
    txt_data = parser.render(data)
    # supprime les apostrophes et le préfixe
    cleaned_text = re.sub(r"[cdjlmnsty]['’](\w*)", r'\1', txt_data, flags=re.IGNORECASE)
    tokens = word_tokenize(cleaned_text, language="french")
    cleaned_tokens = [w.lower() for w in tokens if w.lower()
                      not in french_stop_words and len(w) >= min_length]
    return cleaned_tokens


def load_tokenized_corpus(source: Path, min_length: int = 3, refresh: bool = False) -> list:
    """Load tokenized corpus from file or compute and save it"""
    cfile = Path("corpus-tokens.pkl")
    if cfile.exists() and not refresh:
        with cfile.open("rb") as f:
            return pickle.load(f)
    print("Computing new tokens...")
    corpus = load_raw_corpus(source)
    tokens = []
    bar = ProgressBar(total=len(corpus))
    for slug, doc in bar.iter(corpus):
        token = tokenize(doc)
        if len(token) > min_length:
            tokens.append((slug, token))
    with cfile.open("wb") as f:
        pickle.dump(tokens, f)
    return tokens
