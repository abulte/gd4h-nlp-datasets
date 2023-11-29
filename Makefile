include .env

.PHONY: install
install:
	brew install qsv
	pip install -r requirements.txt
	llm sentence-transformers register distiluse-base-multilingual-cased-v1

.PHONY: download
download:
	wget https://www.data.gouv.fr/fr/datasets/r/f868cca6-8da1-4369-a78d-47463f19a9a3 -O export-dataset.csv

.PHONY: clean
clean:
	llm collections delete $(LLM_COLLECTION) -d $(LLM_DB) || true

.PHONY: filter
QSV_FILTER="archived == 'False' and tonumber(resources_count) > 0 and organization_id"
QSV_COLUMNS=slug,title,description,tags
filter:
	qsv luau filter -d";" $(QSV_FILTER) export-dataset.csv | qsv select $(QSV_COLUMNS) > datasets-filtered.csv

.PHONY: embeddings
embeddings: clean download filter
	llm embed-multi -m $(LLM_MODEL) $(LLM_COLLECTION) datasets-filtered.csv -d $(LLM_DB) --format csv

.PHONY: test
test:
	llm similar $(LLM_COLLECTION) -c 'santé financière entreprises' -d $(LLM_DB)
