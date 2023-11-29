"""Filter learning outputs"""
import csv
from pathlib import Path

from minicli import cli, run

raw_output_path = Path("./output")
output_path = Path("./output-filtered")
datasets_list = Path("export-dataset.csv")
org_mapping = Path(
    "../gd4h/apiextras/apiextras/harvester/data/organizations_mapping.csv"
)


@cli
def map_gd4h(
    number=5000, output_folder: str | None = None, slug_col: str = "slug_or_id"
):
    """
    Parses an NLP output and maps to GD4H organization if any
    """
    output_path.mkdir(exist_ok=True)

    slug_org = []
    with datasets_list.open() as f:
        reader = csv.DictReader(f, delimiter=";")
        for line in reader:
            slug_org.append(
                (line["slug"], line["organization"], line["organization_id"])
            )

    gd4h_map = []
    with org_mapping.open() as f:
        reader = csv.DictReader(f, delimiter=";")
        gd4h_map = [r["datagouvfr_id"] for r in reader if r["datagouvfr_id"]]

    folder = output_folder or "*"
    for nlp_output in raw_output_path.glob(f"{folder}/*.csv"):
        print(f"{nlp_output}...")
        # slug to org info mapping from datasets list
        new_lines = []
        with nlp_output.open() as f:
            reader = csv.DictReader(f)
            for idx, line in enumerate(reader):
                try:
                    ref = next(so for so in slug_org if so[0] == line[slug_col])
                except StopIteration:
                    print(f"[ERROR] slug {line['slug']} not found in catalog")
                    continue
                line["in_gd4h"] = ref[2] in gd4h_map
                new_lines.append(line)
                if idx == number:
                    break

        iter_output_path = output_path / nlp_output.parent.stem
        iter_output_path.mkdir(exist_ok=True)
        iter_output_file = iter_output_path / f"{nlp_output.stem}-mapped.csv"
        with Path(iter_output_file).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=[slug_col, "score", "in_gd4h", "topic"])
            writer.writeheader()
            writer.writerows(new_lines)

        # create a filtered version on in_gd4h
        new_lines = []
        with iter_output_file.open() as f:
            reader = csv.DictReader(f)
            new_lines = [r for r in reader if r["in_gd4h"] == "True"]

        iter_output_file = iter_output_path / f"{nlp_output.stem}-mapped-filtered.csv"
        with Path(iter_output_file).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=[slug_col, "score", "in_gd4h", "topic"])
            writer.writeheader()
            writer.writerows(new_lines)


if __name__ == "__main__":
    run()
