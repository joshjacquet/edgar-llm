import os

os.environ["HF_HUB_OFFLINE"] = "1"

import csv
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from retrieval import retrieve_context, get_collection, build_bm25_index
from models import ExtractedValue
from constants import (
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    OUTPUT_PATH,
    GROUND_TRUTH_PATH,
    PROMPTS_PATH,
)

# GOAL: Use retrieved context + LLM to extract target variable values


def load_llm(model_name=LLM_MODEL_NAME):
    """Load the flan-t5 model and tokenizer."""
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


def extract_value(query, context_chunks, tokenizer, model):
    """
    Given a query and retrieved context chunks, use flan-t5 to extract the answer.
    """
    # Build context from retrieved chunks
    context = "\n\n".join([chunk["chunk_text"] for chunk in context_chunks])

    # Load prompt template from file
    with open(os.path.join(PROMPTS_PATH, "extract.md"), "r") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(context=context, query=query)

    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=32)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer.strip()


def run_extraction(ground_truth_path=GROUND_TRUTH_PATH):
    """
    Run the full extraction pipeline against ground truth queries.
    Returns a list of ExtractedValue results.
    """
    # Load ground truth
    with open(ground_truth_path, "r") as f:
        reader = csv.DictReader(f)
        ground_truth = list(reader)

    # Load models
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Loading LLM...")
    tokenizer, llm = load_llm()

    # Load ChromaDB collection and BM25 index
    json_path = os.path.join(OUTPUT_PATH, "embedded_chunks.json")
    with open(json_path, "r") as f:
        chunks = json.load(f)
    collection = get_collection()
    bm25_data = build_bm25_index([c["chunk_text"] for c in chunks])

    # Run extraction for each ground truth query
    results = []
    for row in ground_truth:
        query = row["query"]
        print(f"  Extracting: {query}")

        context_chunks = retrieve_context(
            query, collection, embed_model, chunks, bm25_data, top_k=10
        )
        extracted = extract_value(query, context_chunks, tokenizer, llm)

        result = ExtractedValue(
            query=query,
            variable=row["variable"],
            expected_value=row["expected_value"],
            extracted_value=extracted,
            section=context_chunks[0]["section"] if context_chunks else "unknown",
        )
        results.append(result)
        print(f"    Expected: {row['expected_value']} | Extracted: {extracted}")

    return results


if __name__ == "__main__":
    results = run_extraction()

    # Write results to output
    output_path = os.path.join(OUTPUT_PATH, "extraction_results.json")
    with open(output_path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)

    print(f"\nWrote {len(results)} results to {output_path}")
