"""
Standalone corpus rebuild: crawls doc.cc.in2p3.fr, chunks, embeds with BGE-M3, and repopulates
the remote Chroma collection. Mirrors notebook 1 (Sections 2-5) exactly, with MAX_PAGES=None
(full site, no cap) -- this is meant for building the REAL corpus (e.g. for the workshop demo
on the server), not the notebook's own quick MAX_PAGES=15 walkthrough.

Run with:  python3 rag_workshop/rebuild_corpus.py   (from the repo root, with the SSH tunnel to
the remote Chroma host open on localhost:8000 -- see notebook 2, Section 9).

Each of the three expensive stages (crawl, chunk, embed) is skipped and loaded from
rag_workshop/corpus/ instead if its output already exists, isn't corrupted, and matches the
current corpus -- so re-running this script after, say, Step 4's Chroma push failed (no tunnel,
timeout, ...) does NOT redo the crawl or the BGE-M3 encoding, it goes straight back to Step 4.
Set FORCE_RECRAWL / FORCE_RECHUNK / FORCE_REEMBED below to force a given stage to redo its work.
"""
import os
import re
import sys
import time
import json
import pickle
import hashlib
import urllib.robotparser as robotparser
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode
from typing import List

import requests
from bs4 import BeautifulSoup
import numpy as np
from tqdm.auto import tqdm

WORKDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKDIR)
from chunk_types import Chunk  # noqa: E402

CORPUS_DIR = os.path.join(WORKDIR, "corpus")
os.makedirs(CORPUS_DIR, exist_ok=True)

SEED_URL = "https://doc.cc.in2p3.fr/"
ALLOWED_DOMAIN = "doc.cc.in2p3.fr"
ALLOWED_PATH_PREFIX = None
USER_AGENT = "AISSAI-RAG-School-Bot/1.0 (educational crawl; contact: your-email@example.org)"
REQUEST_DELAY_SECONDS = 1.0
MAX_PAGES = None  # full site -- see notebook 1, Section 2.3 for the demo-scale (15) alternative
IGNORED_QUERY_KEYS = {"utm_source", "utm_medium", "utm_campaign", "ref", "fbclid"}
HEADERS = {"User-Agent": USER_AGENT}

CHUNK_MAX_WORDS = 180
BATCH_SIZE = 8

# Each stage below is skipped and loaded from rag_workshop/corpus/ instead, if its output file(s)
# already exist, aren't corrupted, and match the current corpus (see the per-stage checks in
# main()). Flip one to True to force that stage to redo its work regardless of what's cached --
# e.g. after editing the crawl/chunk logic itself, or to refresh a corpus that changed upstream.
FORCE_RECRAWL = False
FORCE_RECHUNK = False
FORCE_REEMBED = False

# Never fetch these as "documentation text" -- following a link to one downloads the whole
# binary (this site links multi-hundred-MB software archives under /_downloads/) and then
# tries to parse it as HTML, corrupting the corpus with garbage.
SKIPPED_EXTENSIONS = (
    ".zip", ".tar", ".tar.gz", ".tgz", ".gz", ".rar", ".7z",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".mp4", ".mp3", ".exe", ".dmg", ".iso", ".whl",
)


def is_probably_binary(url: str) -> bool:
    return urlparse(url).path.lower().endswith(SKIPPED_EXTENSIONS)


def can_fetch(url: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    query_pairs = [(k, v) for k, v in parse_qsl(parsed.query) if k not in IGNORED_QUERY_KEYS]
    query_pairs.sort()
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", urlencode(query_pairs), ""))


def in_scope(url: str, allowed_domain: str, allowed_path_prefix) -> bool:
    parsed = urlparse(url)
    if parsed.netloc != allowed_domain:
        return False
    if allowed_path_prefix and not parsed.path.startswith(allowed_path_prefix):
        return False
    return True


def discover_sitemap_urls(seed_url, allowed_domain, allowed_path_prefix):
    parsed_seed = urlparse(seed_url)
    origin = f"{parsed_seed.scheme}://{parsed_seed.netloc}"
    candidate_sitemap_urls = [f"{origin}/sitemap.xml"]
    try:
        robots_resp = requests.get(f"{origin}/robots.txt", headers=HEADERS, timeout=10)
        if robots_resp.ok:
            for line in robots_resp.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    candidate_sitemap_urls.insert(0, line.split(":", 1)[1].strip())
    except Exception:
        pass

    found_urls = []
    for sitemap_url in candidate_sitemap_urls:
        try:
            resp = requests.get(sitemap_url, headers=HEADERS, timeout=10)
            if not resp.ok or "xml" not in resp.headers.get("Content-Type", "") and "<urlset" not in resp.text[:200]:
                continue
            soup = BeautifulSoup(resp.text, "xml")
            locs = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
            found_urls = [u for u in locs if in_scope(u, allowed_domain, allowed_path_prefix)]
            if found_urls:
                print(f"Found sitemap at {sitemap_url}: {len(found_urls)} in-scope URLs.")
                break
        except Exception:
            continue
    if not found_urls:
        print("No usable sitemap found -- will rely on link-following only.")
    return found_urls


def crawl(seed_urls, allowed_domain, max_pages, delay, allowed_path_prefix=None):
    visited = set()
    queue = list(dict.fromkeys(normalize_url(u) for u in seed_urls))
    pages = []

    while queue and (max_pages is None or len(pages) < max_pages):
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        if is_probably_binary(url):
            print(f"skip (binary/download link): {url}")
            continue

        if not can_fetch(url):
            print(f"skip (robots.txt): {url}")
            continue

        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print(f"skip (error): {url} -> {e}")
            continue

        soup = BeautifulSoup(resp.text, "lxml")

        for a in soup.find_all("a", href=True):
            next_url = normalize_url(urljoin(url, a["href"]))
            if (in_scope(next_url, allowed_domain, allowed_path_prefix)
                    and next_url not in visited and not is_probably_binary(next_url)):
                queue.append(next_url)

        for tag in soup(["script", "style", "footer", "header"]):
            tag.decompose()
        for tag in soup.find_all("nav"):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else url
        elements = [
            {"tag": t.name, "text": t.get_text(" ", strip=True)}
            for t in soup.find_all(["h1", "h2", "h3", "p", "li", "pre", "code"])
        ]
        elements = [e for e in elements if e["text"]]
        flat_text = " ".join(e["text"] for e in elements)

        if len(flat_text.split()) > 30:
            pages.append({"url": url, "title": title, "elements": elements, "text": flat_text})
            cap_str = str(max_pages) if max_pages is not None else "?"
            print(f"[{len(pages)}/{cap_str}] fetched: {url}  ({len(flat_text.split())} words, {len(elements)} elements)", flush=True)

        time.sleep(delay)

    return pages


def word_count(text: str) -> int:
    return len(text.split())


def heading_aware_chunk(elements: list, max_words: int = 180) -> List[dict]:
    HEADING_LEVEL = {"h1": 1, "h2": 2, "h3": 3}
    chunks = []
    heading_stack = []
    current_sentences = []
    current_len = 0

    def flush():
        nonlocal current_sentences, current_len
        text = " ".join(current_sentences).strip()
        if text:
            chunks.append({"heading_path": [h[1] for h in heading_stack], "text": text})
        current_sentences, current_len = [], 0

    for el in elements:
        if el["tag"] in HEADING_LEVEL:
            flush()
            level = HEADING_LEVEL[el["tag"]]
            heading_stack = [h for h in heading_stack if h[0] < level]
            heading_stack.append((level, el["text"]))
            continue

        sentences = [el["text"]] if el["tag"] in ("pre", "li") else re.split(r'(?<=[.!?])\s+', el["text"])
        for sent in sentences:
            sent_len = word_count(sent)
            if current_len + sent_len > max_words and current_sentences:
                flush()
            current_sentences.append(sent)
            current_len += sent_len

    flush()
    return chunks


def _load_json_cache(path):
    '''Returns the parsed JSON, or None if the file is missing/corrupted (never raises).'''
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Cached file at {path} is unusable ({e}) -- ignoring it.", flush=True)
        return None


def _load_pickle_cache(path):
    '''Returns the unpickled object, or None if the file is missing/corrupted (never raises).'''
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Cached file at {path} is unusable ({e}) -- ignoring it.", flush=True)
        return None


def _fingerprint(parts) -> str:
    '''A stable content hash over an ordered sequence of strings. Used to validate that a cached
    stage's output still matches its *current* input content -- not just its length or the set of
    URLs it covers, both of which stay identical if a page's text changed without its URL changing
    (e.g. a re-crawl after the site was edited), silently letting a stale cache look "valid".'''
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _pages_fingerprint(pages) -> str:
    ordered = sorted(pages, key=lambda p: p["url"])
    return _fingerprint(f"{p['url']}\x01{p.get('text', '')}" for p in ordered)


def _chunks_fingerprint(chunks) -> str:
    return _fingerprint(f"{c.chunk_id}\x01{c.embed_text}" for c in chunks)


def _read_fingerprint(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def _write_fingerprint(path, value):
    with open(path, "w", encoding="utf-8") as f:
        f.write(value)


def main():
    live_corpus_path = os.path.join(CORPUS_DIR, "corpus_live_sample.json")
    chunks_path = os.path.join(CORPUS_DIR, "chunks.pkl")
    page_text_path = os.path.join(CORPUS_DIR, "page_full_text_by_url.pkl")
    emb_path = os.path.join(CORPUS_DIR, "embeddings.npz")
    lex_path = os.path.join(CORPUS_DIR, "lexical_weights.pkl")
    chunks_fp_path = os.path.join(CORPUS_DIR, "chunks_fingerprint.txt")
    emb_fp_path = os.path.join(CORPUS_DIR, "embeddings_fingerprint.txt")

    print("=== Step 1: crawl ===", flush=True)
    raw_pages = None if FORCE_RECRAWL else _load_json_cache(live_corpus_path)
    if raw_pages:
        print(f"Loaded cached crawl: {len(raw_pages)} pages from {live_corpus_path}.", flush=True)
    else:
        sitemap_urls = discover_sitemap_urls(SEED_URL, ALLOWED_DOMAIN, ALLOWED_PATH_PREFIX)
        seed_urls = sitemap_urls if sitemap_urls else [SEED_URL]
        raw_pages = crawl(seed_urls, ALLOWED_DOMAIN, MAX_PAGES, REQUEST_DELAY_SECONDS, ALLOWED_PATH_PREFIX)
        print(f"\nCrawled {len(raw_pages)} pages.", flush=True)
        with open(live_corpus_path, "w", encoding="utf-8") as f:
            json.dump(raw_pages, f, ensure_ascii=False, indent=2)

    print("\n=== Step 2: chunk ===", flush=True)
    current_pages_fp = _pages_fingerprint(raw_pages)
    all_chunks = None if FORCE_RECHUNK else _load_pickle_cache(chunks_path)
    page_full_text_by_url = None if FORCE_RECHUNK else _load_pickle_cache(page_text_path)
    if all_chunks is not None and page_full_text_by_url is not None:
        cached_pages_fp = _read_fingerprint(chunks_fp_path)
        if cached_pages_fp != current_pages_fp:
            # Content fingerprint, not just URL set or page count: a page can be re-crawled with
            # edited text under the *same* URL, which a same-URLs/same-count check would miss and
            # silently keep serving chunks built from the old text.
            print(
                "Cached chunks don't match the current crawl's content (URLs, text, or count "
                "changed since these were built) -- stale. Re-chunking.",
                flush=True,
            )
            all_chunks = None
    else:
        all_chunks = None

    if all_chunks is not None:
        print(f"Loaded cached chunks: {len(all_chunks)} chunks from {chunks_path}.", flush=True)
    else:
        all_chunks: List[Chunk] = []
        page_full_text_by_url = {}
        for doc_idx, page in enumerate(raw_pages):
            elements = page.get("elements") or [{"tag": "p", "text": page["text"]}]
            page_full_text_by_url[page["url"]] = page.get("text") or " ".join(e["text"] for e in elements)

            pieces = heading_aware_chunk(elements, CHUNK_MAX_WORDS)
            for i, piece in enumerate(pieces):
                heading_path_str = " > ".join(piece["heading_path"])
                header = f"{page['title']} — {heading_path_str}" if heading_path_str else page["title"]
                embed_text = f"{header}\n{piece['text']}"
                all_chunks.append(Chunk(
                    chunk_id=f"doc{doc_idx}_chunk{i}",
                    text=piece["text"],
                    embed_text=embed_text,
                    heading_path=heading_path_str,
                    source_url=page["url"],
                    source_title=page["title"],
                    chunk_index=i,
                ))
        print(f"Total chunks built: {len(all_chunks)} (from {len(raw_pages)} pages)", flush=True)

        with open(chunks_path, "wb") as f:
            pickle.dump(all_chunks, f)
        with open(page_text_path, "wb") as f:
            pickle.dump(page_full_text_by_url, f)
        _write_fingerprint(chunks_fp_path, current_pages_fp)

    print("\n=== Step 3: embed (BGE-M3) ===", flush=True)
    current_chunks_fp = _chunks_fingerprint(all_chunks)
    cached_dense_raw = None
    if not FORCE_REEMBED and os.path.exists(emb_path) and os.path.exists(lex_path):
        try:
            cached_dense_raw = np.load(emb_path)["dense"]
            cached_sparse_raw = _load_pickle_cache(lex_path)
        except Exception as e:
            print(f"Cached file at {emb_path} is unusable ({e}) -- ignoring it.", flush=True)
            cached_dense_raw = None
            cached_sparse_raw = None
        else:
            cached_chunks_fp = _read_fingerprint(emb_fp_path)
            if cached_sparse_raw is None or cached_dense_raw.shape[0] != len(all_chunks) \
                    or len(cached_sparse_raw) != len(all_chunks) \
                    or cached_chunks_fp != current_chunks_fp:
                # Fingerprint catches same-count-but-different-content too (e.g. chunking logic
                # changed, or chunks were rebuilt from different pages but happened to total the
                # same number) -- a pure count comparison would wrongly accept that as valid and
                # pair the new chunks with someone else's dense/sparse vectors.
                print(
                    f"Cached embeddings don't match the current {len(all_chunks)} chunks -- "
                    "stale or partial. Re-embedding.",
                    flush=True,
                )
                cached_dense_raw = None

    if cached_dense_raw is not None:
        dense_embeddings, sparse_weights = cached_dense_raw, cached_sparse_raw
        print(f"Loaded cached embeddings: {dense_embeddings.shape} from {emb_path}.", flush=True)
    else:
        from FlagEmbedding import BGEM3FlagModel
        bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

        all_texts = [c.embed_text for c in all_chunks]
        dense_list, sparse_list = [], []
        for i in tqdm(range(0, len(all_texts), BATCH_SIZE), desc="Encoding chunks"):
            batch = all_texts[i:i + BATCH_SIZE]
            out = bge_model.encode(batch, return_dense=True, return_sparse=True, return_colbert_vecs=False)
            dense_list.append(out["dense_vecs"])
            sparse_list.extend(out["lexical_weights"])
        dense_embeddings = np.vstack(dense_list)
        sparse_weights = sparse_list

        assert dense_embeddings.shape[0] == len(all_chunks)
        assert len(sparse_weights) == len(all_chunks)

        np.savez_compressed(emb_path, dense=dense_embeddings)
        with open(lex_path, "wb") as f:
            pickle.dump(sparse_weights, f)
        _write_fingerprint(emb_fp_path, current_chunks_fp)
        print("Dense embeddings shape:", dense_embeddings.shape, flush=True)

    print("\n=== Step 4: repopulate remote Chroma ===", flush=True)
    import chromadb
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    COLLECTION_NAME = "ccin2p3_docs"
    existing = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME in existing:
        chroma_client.delete_collection(COLLECTION_NAME)
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [c.chunk_id for c in all_chunks]
    documents = [c.text for c in all_chunks]
    metadatas = [
        {
            "source_url": c.source_url,
            "source_title": c.source_title,
            "chunk_index": c.chunk_index,
            "heading_path": c.heading_path,
        }
        for c in all_chunks
    ]
    embeddings_list = dense_embeddings.tolist()

    CHROMA_BATCH_SIZE = 200
    for i in tqdm(range(0, len(ids), CHROMA_BATCH_SIZE), desc="Inserting into Chroma"):
        collection.add(
            ids=ids[i:i + CHROMA_BATCH_SIZE],
            embeddings=embeddings_list[i:i + CHROMA_BATCH_SIZE],
            documents=documents[i:i + CHROMA_BATCH_SIZE],
            metadatas=metadatas[i:i + CHROMA_BATCH_SIZE],
        )
    print(f"Inserted {collection.count()} chunks into remote Chroma.", flush=True)
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
