"""
Standalone, resumable LLM summarization + keyword extraction over a chunked corpus. This is
notebook 1, Section 7.6's slow cell, extracted into its own script -- for a full-size corpus
(hundreds of chunks), running it inline in the notebook kernel ties up that kernel for a long
time and, worse, Ollama's own memory footprint tends to grow across hundreds of sequential calls
until the machine starts swapping (see the periodic unload below). Running it here instead means
you can launch it from a plain terminal, watch it, Ctrl+C it, and resume later without touching
the notebook or losing already-computed work.

Reads:  rag_workshop/corpus/chunks.pkl, rag_workshop/corpus/lexical_weights.pkl
        (built by notebook 1, Sections 3.5 and 4.3, or by rebuild_corpus.py)
Writes: rag_workshop/corpus/chunk_summaries.json (human-readable: chunk id, page title, URL,
        raw text, summary, keywords) and chunk_summaries_embeddings.npz (their embeddings).
        Both are read directly by notebook 1, Section 7.6 -- run this script first, then open the
        notebook and that cell will find everything already cached.

Run with:  python3 rag_workshop/summarize_corpus.py   (from the repo root, with Ollama running
locally and rag_workshop/corpus/chunks.pkl already built)
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import ollama
from tqdm.auto import tqdm

WORKDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKDIR)
from chunk_types import Chunk  # noqa: E402,F401 -- needed to unpickle chunks.pkl

CORPUS_DIR = os.path.join(WORKDIR, "corpus")

GENERATION_MODEL = "llama3.2:3b"
SUMMARY_MAX_CHUNKS = None        # None = the whole corpus; set an int to test on a subset first
KEYWORDS_PER_CHUNK = 8
SUMMARY_INPUT_MAX_CHARS = 4000

# Checkpointing / memory control -- this is the part that matters for a large corpus:
CHECKPOINT_EVERY = 20     # save chunk_summaries.json + .npz to disk every N newly-computed chunks,
                           # so a Ctrl+C, crash, or OOM loses at most this many chunks of work.
UNLOAD_MODEL_EVERY = 50   # every N newly-computed chunks, tag that Ollama call with keep_alive=0
                           # so the server unloads the model right after responding, instead of
                           # keeping accumulating context/state across hundreds of calls -- this
                           # is what caused RAM to climb to several GB and the machine to start
                           # swapping when this ran as one long uninterrupted sequence of calls.

CHUNKS_PATH = os.path.join(CORPUS_DIR, "chunks.pkl")
LEX_PATH = os.path.join(CORPUS_DIR, "lexical_weights.pkl")
SUMMARY_CACHE_JSON = os.path.join(CORPUS_DIR, "chunk_summaries.json")
SUMMARY_CACHE_NPZ = os.path.join(CORPUS_DIR, "chunk_summaries_embeddings.npz")

SUMMARY_PROMPT = '''You are an expert technical-documentation editor and information-retrieval
engineer. Create a faithful, information-dense summary of the passage below for semantic search.
The summary is a retrieval aid, not a replacement for the source: preserve the meaning and do not
invent, generalize, or add facts that are not supported by the passage.

Requirements:
- Write 4 to 8 complete sentences when the passage is substantial; use fewer only when it is truly short.
- Target 90-180 words and never exceed 200 words.
- Preserve the richest semantic context: the topic and scope, the main entities, the user's goal,
  actions and procedures, inputs and outputs, dependencies, conditions, prerequisites, constraints,
  defaults, thresholds, trade-offs, exceptions, warnings, and cause/effect relationships.
- Keep exact technical identifiers verbatim, including commands, flags, parameter names, environment
  variables, API routes, file names, package names, model names, error codes, numeric values, and
  units. Put such identifiers in backticks when natural; never paraphrase them away.
- Prefer concrete, answer-bearing facts and distinctive terminology over generic statements such as
  "this section explains". Preserve the terminology a user would likely use in a question.
- If the passage is a fragment, use its heading or local context to make the summary coherent, but
  do not infer missing details.
- Output only the summary, with no title, bullet list, commentary, or claim of certainty.

Passage:
{chunk_text}

Information-dense summary (4-8 sentences, 90-180 words, maximum 200 words):'''


def summarize_chunk(chunk_text: str, keep_alive=None) -> str:
    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": SUMMARY_PROMPT.format(
            chunk_text=chunk_text[:SUMMARY_INPUT_MAX_CHARS]
        )}],
        options={"temperature": 0.1, "top_p": 0.8, "top_k": 20},
        keep_alive=keep_alive,
    )
    return response["message"]["content"].strip()


def top_keywords_from_sparse(chunk_sparse_weights: dict, bge_model, k: int = KEYWORDS_PER_CHUNK) -> str:
    top_tokens = sorted(chunk_sparse_weights.items(), key=lambda kv: kv[1], reverse=True)[:k]
    words = [bge_model.tokenizer.decode([int(tid)]).strip() for tid, _ in top_tokens]
    return " ".join(w for w in words if w)


def load_summary_cache():
    if not (os.path.exists(SUMMARY_CACHE_JSON) and os.path.exists(SUMMARY_CACHE_NPZ)):
        return {}, {}
    try:
        with open(SUMMARY_CACHE_JSON, encoding="utf-8") as f:
            records = json.load(f)
        npz = np.load(SUMMARY_CACHE_NPZ)
        vectors = {
            chunk_id: (npz["summary_dense"][i], npz["keyword_dense"][i])
            for i, chunk_id in enumerate(npz["chunk_ids"])
        }
    except Exception as e:
        print(f"Found a summary cache, but it's corrupted/unreadable ({e}) -- starting fresh.")
        return {}, {}
    common_ids = set(records) & set(vectors)
    if len(common_ids) != len(records) or len(common_ids) != len(vectors):
        print(
            f"Summary cache has {len(records)} records but {len(vectors)} vectors -- keeping only "
            f"the {len(common_ids)} chunk_ids present in both (likely an interrupted previous run)."
        )
    return (
        {k: v for k, v in records.items() if k in common_ids},
        {k: v for k, v in vectors.items() if k in common_ids},
    )


def save_summary_cache(records, vectors):
    tmp_json = SUMMARY_CACHE_JSON + ".tmp"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp_json, SUMMARY_CACHE_JSON)  # atomic swap -- never leaves a half-written JSON

    chunk_ids = list(vectors.keys())
    tmp_npz = SUMMARY_CACHE_NPZ + ".tmp.npz"
    np.savez_compressed(
        tmp_npz,
        chunk_ids=np.array(chunk_ids),
        summary_dense=np.array([vectors[cid][0] for cid in chunk_ids]),
        keyword_dense=np.array([vectors[cid][1] for cid in chunk_ids]),
    )
    os.replace(tmp_npz, SUMMARY_CACHE_NPZ)


def main():
    with open(CHUNKS_PATH, "rb") as f:
        all_chunks = pickle.load(f)
    with open(LEX_PATH, "rb") as f:
        sparse_weights = pickle.load(f)
    sparse_index = {c.chunk_id: w for c, w in zip(all_chunks, sparse_weights)}

    from FlagEmbedding import BGEM3FlagModel
    bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

    summary_subset = all_chunks[:SUMMARY_MAX_CHUNKS]
    scope_label = "full corpus" if SUMMARY_MAX_CHUNKS is None else f"top {SUMMARY_MAX_CHUNKS} chunks"

    cached_records, cached_vectors = load_summary_cache()
    n_cached = sum(1 for c in summary_subset if c.chunk_id in cached_vectors)
    print(f"Summarizing {len(summary_subset)} of {len(all_chunks)} chunks ({scope_label})... "
          f"{n_cached} already cached.")

    newly_computed = 0
    interrupted = False
    try:
        for c in tqdm(summary_subset, desc="Summarizing + extracting keywords"):
            if c.chunk_id in cached_vectors:
                continue

            force_unload = UNLOAD_MODEL_EVERY and (newly_computed + 1) % UNLOAD_MODEL_EVERY == 0
            summary_text = f"{c.source_title} — {c.heading_path}\n" + summarize_chunk(
                c.text, keep_alive=0 if force_unload else None
            )
            keyword_text = top_keywords_from_sparse(sparse_index[c.chunk_id], bge_model)
            encoded = bge_model.encode([summary_text, keyword_text], return_dense=True)["dense_vecs"]

            cached_records[c.chunk_id] = {
                "chunk_id": c.chunk_id,
                "source_url": c.source_url,
                "source_title": c.source_title,
                "heading_path": c.heading_path,
                "text": c.text,
                "summary": summary_text,
                "keywords": keyword_text,
            }
            cached_vectors[c.chunk_id] = (encoded[0], encoded[1])
            newly_computed += 1

            if CHECKPOINT_EVERY and newly_computed % CHECKPOINT_EVERY == 0:
                save_summary_cache(cached_records, cached_vectors)
                tqdm.write(f"Checkpoint: {newly_computed} new chunks saved to {SUMMARY_CACHE_JSON}")
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted -- saving progress before exiting.")

    save_summary_cache(cached_records, cached_vectors)
    status = "Interrupted after" if interrupted else "Done --"
    print(f"{status} {newly_computed} new chunks computed this run; "
          f"cache now has {len(cached_records)} chunks -> {SUMMARY_CACHE_JSON}")
    if interrupted:
        print("Re-run this script to resume -- already-cached chunks won't be recomputed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
