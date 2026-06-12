import os
import json
import math
from collections import Counter
import chromadb
import snowballstemmer
from sentence_transformers import SentenceTransformer
from constants import EMBEDDING_MODEL_NAME, OUTPUT_PATH

# GOAL: From embedded chunks, retrieve relevant context using hybrid search (vector + BM25)

CHROMA_PATH = os.path.join(OUTPUT_PATH, "chroma_db")
stemmer = snowballstemmer.stemmer("english")


def load_chunks_to_chroma(json_path, collection_name="edgar_chunks"):
    """
    Load embedded chunks from JSON into a ChromaDB collection.
    Returns the collection ready for querying.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection if it exists so we start fresh
    try:
        client.delete_collection(collection_name)
    except (ValueError, chromadb.errors.NotFoundError):
        pass

    collection = client.create_collection(name=collection_name)

    with open(json_path, "r") as f:
        chunks = json.load(f)

    ids = [str(i) for i in range(len(chunks))]
    documents = [chunk["chunk_text"] for chunk in chunks]
    embeddings = [chunk["embeddings"] for chunk in chunks]
    metadatas = [
        {"cik": chunk["cik"], "section": chunk["section"], "year": chunk["year"]}
        for chunk in chunks
    ]

    # ChromaDB has a batch limit, so add in batches
    batch_size = 5000
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"Loaded {len(chunks)} chunks into ChromaDB collection '{collection_name}'")
    return collection


def get_collection(collection_name="edgar_chunks"):
    """
    Get an existing ChromaDB collection.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=collection_name)


# --- BM25 helpers ---


def tokenize(text):
    """Lowercase, split, and stem."""
    words = text.lower().split()
    return stemmer.stemWords(words)


def build_bm25_index(documents):
    """
    Pre-compute BM25 stats from a list of document strings.
    Returns tokenized docs, average doc length, and document frequency map.
    """
    tokenized = [tokenize(doc) for doc in documents]
    avg_dl = sum(len(d) for d in tokenized) / len(tokenized)

    # Document frequency: how many docs contain each term
    df_map = Counter()
    for doc in tokenized:
        unique_terms = set(doc)
        for term in unique_terms:
            df_map[term] += 1

    return tokenized, avg_dl, df_map


def bm25_score(query_tokens, doc_tokens, avg_dl, doc_count, df_map, k1=1.5, b=0.75):
    """BM25 relevance score for a single document."""
    score = 0.0
    dl = len(doc_tokens)
    for term in query_tokens:
        tf = doc_tokens.count(term)
        df = df_map.get(term, 0)
        idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
    return score


def bm25_search(query, documents, tokenized_docs, avg_dl, df_map, top_k=10):
    """
    Score all documents against a query using BM25.
    Returns list of (doc_index, score) sorted by score descending.
    """
    query_tokens = tokenize(query)
    doc_count = len(documents)

    scores = []
    for i, doc_tokens in enumerate(tokenized_docs):
        score = bm25_score(query_tokens, doc_tokens, avg_dl, doc_count, df_map)
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# --- Hybrid search with RRF ---


def reciprocal_rank_fusion(rankings, weights=None, k=60):
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    Each ranking is a list of doc IDs ordered by relevance.
    Weights control relative importance of each ranking (default: equal).
    """
    if weights is None:
        weights = [1.0] * len(rankings)

    fused_scores = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + weight / (k + rank + 1)

    return sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)


def retrieve_context(
    query, collection, model, chunks, bm25_data, top_k=5, weights=None
):
    """
    Hybrid retrieval: vector search (ChromaDB) + BM25, combined with RRF.
    Returns a list of dicts with chunk_text and section.
    """
    # Dense vector search
    query_embedding = model.encode(query).tolist()
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2,
    )
    vector_ranking = [int(doc_id) for doc_id in vector_results["ids"][0]]

    # BM25 search
    tokenized_docs, avg_dl, df_map = bm25_data
    documents = [c["chunk_text"] for c in chunks]
    bm25_results = bm25_search(
        query, documents, tokenized_docs, avg_dl, df_map, top_k=top_k * 2
    )
    bm25_ranking = [idx for idx, score in bm25_results]

    # Combine with RRF (weights: [vector_weight, bm25_weight])
    fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking], weights=weights)
    top_ids = fused[:top_k]

    # Fetch the actual documents
    results = collection.get(
        ids=[str(i) for i in top_ids], include=["documents", "metadatas"]
    )

    context_chunks = []
    for i in range(len(results["ids"])):
        context_chunks.append(
            {
                "chunk_text": results["documents"][i],
                "section": results["metadatas"][i]["section"],
            }
        )

    return context_chunks


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"

    json_path = os.path.join(OUTPUT_PATH, "embedded_chunks.json")

    # Load chunks into ChromaDB
    collection = load_chunks_to_chroma(json_path)

    # Build BM25 index from the same chunks
    with open(json_path, "r") as f:
        chunks = json.load(f)
    documents = [c["chunk_text"] for c in chunks]
    bm25_data = build_bm25_index(documents)

    # Test a retrieval
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    results = retrieve_context(
        "What was AIG's total revenue in 2020?",
        collection,
        model,
        chunks,
        bm25_data,
    )

    print(f"\nTop {len(results)} results (hybrid search):")
    for r in results:
        print(f"  [{r['section']}]")
        print(f"  {r['chunk_text'][:200]}...")
        print()
