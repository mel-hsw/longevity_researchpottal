PYTHON = python3
export KMP_DUPLICATE_LIB_OK=TRUE

.PHONY: setup manifest ingest query eval outputs portal clean

setup:
	pip install -r requirements.txt
	mkdir -p data/raw data/processed logs reports outputs outputs/threads outputs/artifacts
	@if [ ! -f data/raw/Barry2014.pdf ] && [ -d ../Paper ]; then \
		echo "Copying PDFs from ../Paper/ to data/raw/..."; \
		cp ../Paper/*.pdf data/raw/; \
	fi
	@echo "Setup complete.  Add your OPENAI_API_KEY to .env"

portal:
	streamlit run src/app/main.py

manifest:
	$(PYTHON) -m src.ingest.build_manifest

ingest:
	$(PYTHON) -m src.ingest.run_ingest

query:
	@if [ -z "$(QUERY)" ]; then \
		echo "Usage: make query QUERY=\"your question here\""; \
		exit 1; \
	fi
	$(PYTHON) scripts/query.py "$(QUERY)"

eval:
	$(PYTHON) -m src.eval.run_eval

outputs:
	$(PYTHON) scripts/generate_outputs.py

clean:
	rm -rf data/processed/faiss_index data/processed/bm25_index.pkl
	rm -f data/processed/chunks.jsonl
	rm -f logs/run_log.jsonl
