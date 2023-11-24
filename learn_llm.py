"""Use llm toolkit embeddings (not using a LLM model!)
https://llm.datasette.io/en/stable/embeddings/index.html"""

import csv
import os

from datetime import datetime
from pathlib import Path

import llm
import sqlite_utils

from dotenv import load_dotenv
from minicli import cli, run
from slugify import slugify

from utils import load_raw_keywords, load_kw_file

load_dotenv()

db = sqlite_utils.Database(os.getenv("LLM_DB"))
collection = llm.Collection(os.getenv("LLM_COLLECTION"), db)


@cli
def compute(number: int = 100):
    output_path = Path("./output") / datetime.now().strftime('%Y%m%d-%H%M%S')
    output_path.mkdir()
    model_name = collection.model_id.split("/")[1]
    for kw in [*load_kw_file().keys(), "all"]:
        print(f'Computing for theme "{kw}"...')
        corpus = load_raw_keywords(kw)
        theme = slugify(kw)
        entries = collection.similar(" ".join(corpus), number=number)
        with (output_path / f"{model_name}-output_{theme}.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=["slug", "score"])
            writer.writerows([
                {
                    "slug": entry.id,
                    "score": entry.score
                } for entry in entries
            ])


if __name__ == "__main__":
    run()
