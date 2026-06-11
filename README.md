# edgar-llm
RAG pipeline for extracting structured data from SEC 10-K filings (EDGAR corpus).

## Approach & Pipeline
1. **Ingest** (`src/ingest.py`) -- Pulls a target 10-K from the HuggingFace EDGAR corpus, splits sections into chunks via LangChain, and generates embeddings with sentence-transformers.
2. **Retrieve** (`src/retrieval.py`) -- Hybrid search combining dense vector search (ChromaDB) with BM25 keyword scoring, merged via Reciprocal Rank Fusion (RRF).
3. **Extract** (`src/extract.py`) -- Feeds retrieved context chunks to flan-t5 and prompts it to extract specific values (revenue, net income, etc).
4. **Score** (`src/scoring.py`) -- Compares extracted values against ground truth (`data/validation/ground_truth.csv`). Numeric scoring uses tolerance-based matching, categorical uses exact match.

### Other files
- `src/config.py` -- Spark session config
- `src/constants.py` -- Model names, target company (AIG), paths
- `src/models.py` -- Pydantic models for extraction results

## Results

### Analysis
Results are poor. Numeric values passing at 20%.
Root causes:
1) **Retrieval**. Low power local embeddings models, combined with limited data cleaning.
    - Future work: Improve data processing logic to handle tables; Leverage larger or more specialized embeddings models.
2) **Instruction following**. Expectations missed when expecting to pull numeric data and receiving sentence instead.
    - Future work: Improve retrieval for more specific context; tweak prompting; leverage models which support structured outputs.

### Result Table
| Variable | Match | Percent |
| --- | --- | --- |
| Total Revenues | 1/5 | 20% |
| Net Income | 1/5 | 20% |
| Stock Exchange | 5/5 | 100% |
| **Total** | **7/15** | **46.7%** |


## Setup
```bash
pip install -r requirements.txt
```
Models need to be cached locally before running offline. This was done for local Spark issues. Use `hf download <model_name>` to pull them down.

## Running
From the `src/` directory:
```bash
python ingest.py       # fetch, chunk, embed -> data/output/embedded_chunks.json
python retrieval.py    # load into ChromaDB + test query
python extract.py      # run extraction against ground truth
python scoring.py      # score results
```

## Constraints
- Company: AIG (CIK 5272)
- Filing: 10-K, 2020
- Dataset: [eloukas/edgar-corpus](https://huggingface.co/datasets/eloukas/edgar-corpus)
- Language: Python / PySpark

## Formatting
`ruff` -- `ruff_check.bat` provides a quick formatting & linting pass.
