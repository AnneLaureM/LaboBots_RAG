"""
Standalone DENSE-ONLY RAG pipeline: crawl -> intelligent (heading-aware) chunking -> dense-only
BGE-M3 embeddings -> push to the remote Chroma collection -> a quick benchmark + an interactive
search loop to check retrieval quality and latency.

WHY THIS SCRIPT EXISTS (vs. rag_workshop/rebuild_corpus.py):
rebuild_corpus.py builds a HYBRID (dense + sparse) index with BGE-M3. This script builds the same
corpus but skips the sparse half entirely (return_sparse=False) -- a simpler, dense-only index.
It still uses BGE-M3 (not a different "pure dense" model): streamlit_app.py and
streamlit_app_secure.py hardcode BGE-M3 to encode the user's question at query time, so the
ingestion side must keep using the same model, or the stored vectors and the query vector would no
longer live in the same embedding space and retrieval would silently return wrong results (not an
error -- just bad matches). See README.md, "Dense-only RAG pipeline", for the full rationale.

PREREQUISITES:
    uv sync --extra embeddings

FULL TERMINAL TIMELINE FOR THIS SCRIPT:
    1. In another terminal, BEFORE running this script:
           ./rag_workshop/manage_remote_rag.sh tunnel
       (opens the SSH tunnel this script needs for its Chroma push -- Step 4 below)
    2. Run this script:
           uv run python3 rag_workshop/dense_rag_pipeline.py
    3. Only needed once per workshop, usually already done earlier -- check the remote
       Ollama/LiteLLM stack is up before using the Streamlit apps' chat feature:
           ./rag_workshop/manage_litellm.sh status
    4. Launch the app that will actually chat using the index this script just built:
           streamlit run rag_workshop/streamlit_app.py
           # or, with per-participant login:
           streamlit run rag_workshop/streamlit_app_secure.py

This script never calls an LLM itself -- only retrieval (Steps 1-5 below). Generation (Ollama,
behind the remote LiteLLM proxy) only happens later, inside the Streamlit apps.
"""
import os
import sys
import time
import json
import pickle
from typing import List

import numpy as np
from tqdm.auto import tqdm

WORKDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKDIR)
from chunk_types import Chunk  # noqa: E402

# Reuse the existing scrape/chunk stages instead of duplicating ~250 lines of crawling and
# chunking logic -- they don't depend on dense-vs-hybrid at all, only the embedding step does.
from rebuild_corpus import (  # noqa: E402
    SEED_URL,
    ALLOWED_DOMAIN,
    ALLOWED_PATH_PREFIX,
    MAX_PAGES,
    REQUEST_DELAY_SECONDS,
    CHUNK_MAX_WORDS,
    crawl,
    discover_sitemap_urls,
    heading_aware_chunk,
    _load_json_cache,
    _load_pickle_cache,
)

CORPUS_DIR = os.path.join(WORKDIR, "corpus")
os.makedirs(CORPUS_DIR, exist_ok=True)

# Paths shared with rebuild_corpus.py / the Streamlit apps -- these are fixed, hardcoded paths on
# the app side, so this script must write to the exact same places for "just launch streamlit_app.py
# afterward" to work without editing secrets.toml.
LIVE_CORPUS_PATH = os.path.join(CORPUS_DIR, "corpus_live_sample.json")
CHUNKS_PATH = os.path.join(CORPUS_DIR, "chunks.pkl")
PAGE_TEXT_PATH = os.path.join(CORPUS_DIR, "page_full_text_by_url.pkl")
LEXICAL_WEIGHTS_PATH = os.path.join(CORPUS_DIR, "lexical_weights.pkl")
# rebuild_corpus.py's own cache-staleness marker for lexical_weights.pkl -- removed by this script
# (see write_empty_sparse_cache) so a later rebuild_corpus.py run doesn't mistake an empty,
# dense-only sparse cache for a valid hybrid one and skip recomputing real sparse weights.
EMB_FINGERPRINT_PATH = os.path.join(CORPUS_DIR, "embeddings_fingerprint.txt")
# This script's OWN cache file, separate from rebuild_corpus.py's embeddings.npz, so the two
# scripts' caches never interfere with each other.
DENSE_EMB_PATH = os.path.join(CORPUS_DIR, "dense_embeddings.npz")

BATCH_SIZE = 8
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
CHROMA_COLLECTION_NAME = "ccin2p3_docs"  # same collection the Streamlit apps already query
CHROMA_BATCH_SIZE = 200

# This script favors a simple "does the cache file exist" check over rebuild_corpus.py's
# content-fingerprint staleness detection, to stay easy to read end to end. Delete the relevant
# file under rag_workshop/corpus/ (or flip a flag here to True) to force that stage to redo its
# work.
FORCE_RECRAWL = False
FORCE_RECHUNK = False
FORCE_REEMBED = False

# A couple of example questions to sanity-check retrieval quality/latency at a glance on every
# run -- edit freely once you know what's actually in the corpus. `expected_substring` just needs
# to appear somewhere in a genuinely correct answer chunk (same idea as notebook 1, Section 7.6).
EVAL_QUESTIONS = [
    ("How do I submit a job with SLURM?", "sbatch"),
]


# --------------------------------------------------------------------------
# Step 1: scraping
# --------------------------------------------------------------------------

def crawl_site():
    raw_pages = None if FORCE_RECRAWL else _load_json_cache(LIVE_CORPUS_PATH)
    if raw_pages:
        print(f"Loaded cached crawl: {len(raw_pages)} pages from {LIVE_CORPUS_PATH}.", flush=True)
        return raw_pages

    sitemap_urls = discover_sitemap_urls(SEED_URL, ALLOWED_DOMAIN, ALLOWED_PATH_PREFIX)
    seed_urls = sitemap_urls if sitemap_urls else [SEED_URL]
    raw_pages = crawl(seed_urls, ALLOWED_DOMAIN, MAX_PAGES, REQUEST_DELAY_SECONDS, ALLOWED_PATH_PREFIX)
    print(f"\nCrawled {len(raw_pages)} pages.", flush=True)

    with open(LIVE_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_pages, f, ensure_ascii=False, indent=2)
    return raw_pages


# --------------------------------------------------------------------------
# Step 2: "intelligent" chunking
#
# heading_aware_chunk() never lets a chunk cross an h1/h2/h3 boundary, and packs sentences up to
# CHUNK_MAX_WORDS -- as opposed to a naive fixed-size split, which would cut sentences or whole
# sections in half. See notebook 1, Section 3, for the from-scratch walkthrough (fixed-size ->
# structure-aware -> heading-aware).
# --------------------------------------------------------------------------

def chunk_pages(raw_pages):
    cached_chunks = None if FORCE_RECHUNK else _load_pickle_cache(CHUNKS_PATH)
    cached_page_text = None if FORCE_RECHUNK else _load_pickle_cache(PAGE_TEXT_PATH)
    if cached_chunks is not None and cached_page_text is not None:
        print(f"Loaded cached chunks: {len(cached_chunks)} chunks from {CHUNKS_PATH}.", flush=True)
        return cached_chunks

    chunks: List[Chunk] = []
    page_full_text_by_url = {}
    for doc_idx, page in enumerate(raw_pages):
        elements = page.get("elements") or [{"tag": "p", "text": page["text"]}]
        page_full_text_by_url[page["url"]] = page.get("text") or " ".join(e["text"] for e in elements)

        pieces = heading_aware_chunk(elements, CHUNK_MAX_WORDS)
        for i, piece in enumerate(pieces):
            heading_path_str = " > ".join(piece["heading_path"])
            header = f"{page['title']} — {heading_path_str}" if heading_path_str else page["title"]
            chunks.append(Chunk(
                chunk_id=f"doc{doc_idx}_chunk{i}",
                text=piece["text"],
                embed_text=f"{header}\n{piece['text']}",
                heading_path=heading_path_str,
                source_url=page["url"],
                source_title=page["title"],
                chunk_index=i,
            ))
    print(f"Total chunks built: {len(chunks)} (from {len(raw_pages)} pages)", flush=True)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    with open(PAGE_TEXT_PATH, "wb") as f:
        pickle.dump(page_full_text_by_url, f)
    return chunks


# --------------------------------------------------------------------------
# Step 3: dense-only embeddings
# --------------------------------------------------------------------------

def embed_chunks_dense(chunks):
    if not FORCE_REEMBED and os.path.exists(DENSE_EMB_PATH):
        dense_embeddings = np.load(DENSE_EMB_PATH)["dense"]
        if dense_embeddings.shape[0] == len(chunks):
            print(f"Loaded cached dense embeddings: {dense_embeddings.shape} from {DENSE_EMB_PATH}.", flush=True)
            return dense_embeddings
        print("Cached dense embeddings don't match the current chunk count -- re-embedding.", flush=True)

    from FlagEmbedding import BGEM3FlagModel
    bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

    all_texts = [c.embed_text for c in chunks]
    dense_list = []
    for i in tqdm(range(0, len(all_texts), BATCH_SIZE), desc="Encoding chunks (dense only)"):
        batch = all_texts[i:i + BATCH_SIZE]
        out = bge_model.encode(batch, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        dense_list.append(out["dense_vecs"])
    dense_embeddings = np.vstack(dense_list)
    assert dense_embeddings.shape[0] == len(chunks)

    np.savez_compressed(DENSE_EMB_PATH, dense=dense_embeddings)
    print("Dense embeddings shape:", dense_embeddings.shape, flush=True)
    return dense_embeddings


def write_empty_sparse_cache(num_chunks):
    """
    Always (re)written, regardless of whether dense embeddings came from cache: keeps
    lexical_weights.pkl in sync with the current chunk count and keeps the Streamlit apps
    permanently in dense-only mode no matter which script ran last (this one, or
    rebuild_corpus.py). With every chunk's sparse weights empty, the apps' Reciprocal Rank Fusion
    step naturally degrades to pure dense ranking -- exactly the point of this script, with zero
    changes needed in streamlit_app.py / streamlit_app_secure.py.
    """
    with open(LEXICAL_WEIGHTS_PATH, "wb") as f:
        pickle.dump([{} for _ in range(num_chunks)], f)
    if os.path.exists(EMB_FINGERPRINT_PATH):
        os.remove(EMB_FINGERPRINT_PATH)


# --------------------------------------------------------------------------
# Step 4: vector store (remote Chroma)
# --------------------------------------------------------------------------

def push_to_chroma(chunks, dense_embeddings):
    import chromadb
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    existing = [c.name for c in chroma_client.list_collections()]
    if CHROMA_COLLECTION_NAME in existing:
        chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
    collection = chroma_client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "source_url": c.source_url,
            "source_title": c.source_title,
            "chunk_index": c.chunk_index,
            "heading_path": c.heading_path,
        }
        for c in chunks
    ]
    embeddings_list = dense_embeddings.tolist()

    for i in tqdm(range(0, len(ids), CHROMA_BATCH_SIZE), desc="Inserting into Chroma"):
        collection.add(
            ids=ids[i:i + CHROMA_BATCH_SIZE],
            embeddings=embeddings_list[i:i + CHROMA_BATCH_SIZE],
            documents=documents[i:i + CHROMA_BATCH_SIZE],
            metadatas=metadatas[i:i + CHROMA_BATCH_SIZE],
        )
    print(f"Inserted {collection.count()} chunks into remote Chroma ('{CHROMA_COLLECTION_NAME}').", flush=True)
    return collection


# --------------------------------------------------------------------------
# Step 5: test the vector store -- a small automatic benchmark, then an interactive search loop.
# Retrieval only, no LLM call: generation happens later, in the Streamlit apps (see module
# docstring).
# --------------------------------------------------------------------------

def search(query, collection, bge_model, top_k=5):
    """Dense-only retrieval: embed the query with the same BGE-M3 dense branch used for
    ingestion, query Chroma, and report latency alongside the results."""
    t0 = time.perf_counter()
    query_vec = bge_model.encode([query], return_dense=True, return_sparse=False)["dense_vecs"][0]
    result = collection.query(query_embeddings=[query_vec.tolist()], n_results=top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    hits = []
    for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        # The collection was created with metadata={"hnsw:space": "cosine"}, so Chroma's
        # "distance" here is (1 - cosine similarity); convert back to the more familiar score.
        hits.append({
            "text": doc,
            "source_title": meta["source_title"],
            "source_url": meta["source_url"],
            "score": 1 - distance,
        })
    return hits, elapsed_ms


def run_benchmark(collection, bge_model):
    print(f"\n--- Quick benchmark ({len(EVAL_QUESTIONS)} question(s), dense-only retrieval) ---", flush=True)
    print(f"{'Question':<45} {'Hit@1':>6} {'Hit@3':>6} {'Latency (ms)':>13}")
    for question, expected_substring in EVAL_QUESTIONS:
        hits, elapsed_ms = search(question, collection, bge_model, top_k=3)
        rank = next(
            (i for i, h in enumerate(hits, start=1) if expected_substring.lower() in h["text"].lower()),
            None,
        )
        hit1 = "yes" if rank == 1 else "no"
        hit3 = "yes" if rank is not None else "no"
        label = question if len(question) <= 45 else question[:42] + "..."
        print(f"{label:<45} {hit1:>6} {hit3:>6} {elapsed_ms:>13.1f}")


def interactive_search(collection, bge_model):
    print("\n--- Interactive search (dense-only) -- type a question, empty line to quit ---", flush=True)
    while True:
        try:
            query = input("\n> ").strip()
        except EOFError:
            break
        if not query:
            break
        hits, elapsed_ms = search(query, collection, bge_model, top_k=5)
        for rank, hit in enumerate(hits, start=1):
            preview = hit["text"][:200].replace("\n", " ")
            print(f"  [{rank}] score={hit['score']:.3f}  {hit['source_title']} ({hit['source_url']})")
            print(f"       {preview}...")
        print(f"  -> {len(hits)} result(s) in {elapsed_ms:.1f} ms")


def main():
    print("=== Step 1: crawl ===", flush=True)
    raw_pages = crawl_site()

    print("\n=== Step 2: intelligent (heading-aware) chunking ===", flush=True)
    chunks = chunk_pages(raw_pages)

    print("\n=== Step 3: dense-only embeddings (BGE-M3) ===", flush=True)
    dense_embeddings = embed_chunks_dense(chunks)
    write_empty_sparse_cache(len(chunks))

    # Terminal (in another window, BEFORE this point if you haven't already):
    #   ./rag_workshop/manage_remote_rag.sh tunnel
    print("\n=== Step 4: push to remote Chroma ===", flush=True)
    collection = push_to_chroma(chunks, dense_embeddings)

    print("\n=== Step 5: test retrieval (benchmark + interactive) ===", flush=True)
    from FlagEmbedding import BGEM3FlagModel
    bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    run_benchmark(collection, bge_model)
    interactive_search(collection, bge_model)

    print("\n=== DONE ===", flush=True)
    print(
        "\nThis script only tested retrieval -- no LLM was called. To chat with an LLM grounded in "
        "this dense-only index:\n"
        "\n"
        "  Terminal -- make sure the remote LiteLLM/Ollama stack is up (usually already done\n"
        "  earlier in the workshop):\n"
        "      ./rag_workshop/manage_litellm.sh status\n"
        "\n"
        "  Terminal -- launch the chat app:\n"
        "      streamlit run rag_workshop/streamlit_app.py\n"
        "      # or, with per-participant login:\n"
        "      streamlit run rag_workshop/streamlit_app_secure.py\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
