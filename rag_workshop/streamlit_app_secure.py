"""
Secured, multi-user, themed distributed hybrid RAG chat client.
Run with:  streamlit run rag_workshop/streamlit_app_secure.py
Configuration lives in rag_workshop/.streamlit/secrets.toml and config.toml (Section 14).

This is Section 12's streamlit_app.py plus three additions: an auth gate (OIDC or demo,
Section 14.3), a per-user LiteLLM key lookup (14.1) instead of one shared key, and a themed
option-menu sidebar (14.2). The retrieval/generation pipeline itself is untouched.
"""
import logging
import pickle
import numpy as np
import streamlit as st
import chromadb
from FlagEmbedding import BGEM3FlagModel
from streamlit_option_menu import option_menu
import requests
from chunk_types import Chunk  # noqa: F401 -- needed to unpickle chunks.pkl (see notebook 1, 3.3)

# Streamlit's file watcher walks every loaded module (via transformers, pulled in by
# BGEM3FlagModel) to find source files to watch, which lazy-imports transformers' optional
# vision submodules -- several of those require torchvision (a normal dependency here, see
# pyproject.toml's "embeddings" extra). Belt-and-suspenders only: if an environment hasn't been
# resynced (`uv sync --extra embeddings`) and torchvision is genuinely missing, this just stops
# Streamlit from spamming a full traceback per submodule instead of the app failing to start.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# --------------------------------------------------------------------------
# Configuration -- loaded from .streamlit/secrets.toml, nothing hardcoded here
# --------------------------------------------------------------------------
REQUIRED_SECRETS = ["chroma_host", "chroma_port", "chroma_collection_name",
                     "litellm_proxy_url", "litellm_model_name", "auth_mode"]
missing = [k for k in REQUIRED_SECRETS if k not in st.secrets]
if missing:
    st.error(
        f"Missing secrets: {missing}. Fill in rag_workshop/.streamlit/secrets.toml "
        "(see notebook 2, Section 14.1) before running this app."
    )
    st.stop()

CHROMA_HOST = st.secrets["chroma_host"]
CHROMA_PORT = st.secrets["chroma_port"]
CHROMA_COLLECTION_NAME = st.secrets["chroma_collection_name"]

LITELLM_PROXY_URL = st.secrets["litellm_proxy_url"]
LITELLM_MODEL_NAME = st.secrets["litellm_model_name"]
AUTH_MODE = st.secrets["auth_mode"]  # "oidc" or "demo" -- see Section 14.0

LEXICAL_WEIGHTS_PATH = "rag_workshop/corpus/lexical_weights.pkl"
CHUNKS_PATH = "rag_workshop/corpus/chunks.pkl"
PAGE_TEXT_PATH = "rag_workshop/corpus/page_full_text_by_url.pkl"

EXPAND_TOP_N_TO_FULL_PAGE = 1
MAX_EXPANDED_CHARS = 3000

# How many previous (user, assistant) turns to resend to the LLM as conversation context, so
# follow-up questions ("and for GPU jobs?") work. Kept small on purpose: each turn adds tokens
# to every subsequent call, which costs against the participant's own max_budget (Section 10.4).
MAX_HISTORY_TURNS = 3

# --------------------------------------------------------------------------
# Confidence bands -- ports notebook 1, Section 7.1 to the distributed client. The more we
# trust the retrieved evidence, the less "creative freedom" the model gets; below the lowest
# band there's no usable evidence at all, so the model answers from general knowledge instead,
# clearly flagged as such rather than just refusing outright.
# --------------------------------------------------------------------------
RAG_ONLY_THRESHOLD = 0.80
RAG_SYNTHESIS_THRESHOLD = 0.60
RAG_HEDGE_THRESHOLD = 0.50

# A chunk pulled into the fused top-k isn't necessarily one the model actually leaned on --
# RRF can include a sparse-only match with no dense score at all (cosine_similarity=None), or a
# weak dense hit, purely on lexical overlap. Only link sources with an individually solid
# cosine, deduplicated by page.
SOURCE_LINK_MIN_COSINE = 0.60
MAX_SOURCE_LINKS = 10

RAG_SYSTEM_PROMPT = (
    "You are a helpful technical assistant for a research computing center. "
    "Answer ONLY using the information in the provided context chunks. If the context does "
    "not contain the answer, say so explicitly instead of guessing. Always mention which "
    "source(s) (by title) you used. Keep answers concise and technically precise. Respond in "
    "the same language as the user's question. Earlier turns in the conversation may be "
    "included for context (e.g. a follow-up question) -- ground every factual claim in the "
    "context chunks provided with THIS question, not in what was said earlier."
)

LLM_ONLY_SYSTEM_PROMPT = (
    "You are a helpful technical assistant. No relevant passage was found in the lab's "
    "documentation for this question, so answer from your general knowledge instead. You MUST "
    "start your answer with an explicit note that this is general knowledge, not verified "
    "against the lab's own documentation, and that the user should double-check anything "
    "specific to their cluster/site. Respond in the same language as the user's question."
)

# Label + accent color per mode, used for the little badge shown above each answer.
MODE_STYLE = {
    "rag_only":      {"label": "Documentation — high confidence",   "color": "#16A34A"},
    "rag_synthesis": {"label": "Documentation + light synthesis",        "color": "#2563EB"},
    "rag_hedged":    {"label": "Documentation — low confidence",    "color": "#D97706"},
    "llm_only":      {"label": "General knowledge — not in docs",   "color": "#7C3AED"},
}


def sampling_for_similarity(top_similarity):
    '''Map a top cosine similarity to (mode_name, sampling_options) -- see notebook 1, 7.1.'''
    if top_similarity >= RAG_ONLY_THRESHOLD:
        return "rag_only", {"temperature": 0.1, "top_p": 0.7, "top_k": 10}
    elif top_similarity >= RAG_SYNTHESIS_THRESHOLD:
        return "rag_synthesis", {"temperature": 0.3, "top_p": 0.7, "top_k": 10}
    elif top_similarity >= RAG_HEDGE_THRESHOLD:
        return "rag_hedged", {"temperature": 0.7, "top_p": 0.7, "top_k": 10}
    else:
        return "llm_only", {"temperature": 0.7, "top_p": 0.9, "top_k": 40}


HERO_CSS = """
<style>
.hero {
    background: linear-gradient(120deg, #0D9488 0%, #2563EB 100%);
    padding: 1.4rem 1.8rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.2rem;
}
.hero h1 { margin: 0; font-size: 1.5rem; }
.hero p { margin: 0.3rem 0 0 0; opacity: 0.92; font-size: 0.92rem; }
.mode-badge {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.sources-box {
    background: #F0FDFA;
    border-left: 3px solid #0D9488;
    padding: 0.6rem 0.9rem;
    border-radius: 8px;
    margin-top: 0.5rem;
    font-size: 0.9rem;
}
</style>
"""


def mode_badge(mode):
    style = MODE_STYLE.get(mode)
    if not style:
        return
    st.markdown(
        f'<span class="mode-badge" style="background:{style["color"]}22; color:{style["color"]};">'
        f'{style["label"]}</span>',
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Lab RAG Assistant (secured)", page_icon="🔐", layout="wide")
st.markdown(HERO_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 14.3.1 -- Auth gate: nothing below this block runs for an unverified visitor
# --------------------------------------------------------------------------

def require_login_oidc():
    """Production path: delegate identity entirely to the institution's OIDC provider."""
    if not st.user.is_logged_in:
        st.markdown(
            '<div class="hero"><h1>🔐 Lab RAG Assistant</h1>'
            '<p>Please sign in with your institutional account to continue.</p></div>',
            unsafe_allow_html=True,
        )
        st.button("Log in", on_click=st.login)
        st.stop()
    return st.user.get("name", st.user.email), st.user.email


def require_login_demo():
    """Workshop-only path: a small username/password list -- see the warning in 14.0."""
    import streamlit_authenticator as stauth

    demo_cfg = st.secrets["demo_auth"]
    credentials = {"usernames": {
        uname: dict(udata) for uname, udata in demo_cfg["credentials"]["usernames"].items()
    }}
    authenticator = stauth.Authenticate(
        credentials, demo_cfg["cookie_name"], demo_cfg["cookie_key"], demo_cfg["cookie_expiry_days"],
    )
    try:
        authenticator.login()
    except Exception as e:
        st.error(f"Login error: {e}")
        st.stop()

    status = st.session_state.get("authentication_status")
    if status is False:
        st.error("Incorrect username or password.")
        st.stop()
    if status is None:
        st.markdown(
            '<div class="hero"><h1>🔐 Lab RAG Assistant (workshop demo login)</h1>'
            '<p>Please enter your username and password.</p></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    username = st.session_state["username"]
    name = st.session_state["name"]
    email = credentials["usernames"][username]["email"]
    st.session_state["_authenticator"] = authenticator  # kept for the logout button in the sidebar
    return name, email


if AUTH_MODE == "oidc":
    USER_NAME, USER_EMAIL = require_login_oidc()
elif AUTH_MODE == "demo":
    USER_NAME, USER_EMAIL = require_login_demo()
else:
    st.error(f"Unknown auth_mode {AUTH_MODE!r} in secrets.toml -- expected 'oidc' or 'demo'.")
    st.stop()


def get_user_litellm_key(email: str) -> str:
    """
    Per-user LiteLLM key lookup (14.1) -- replaces Section 12's single shared key. Falls back
    to secrets['litellm_key'] (if set) so the app still runs for an identified-but-unregistered
    visitor, clearly labelled as a fallback rather than failing outright.
    """
    user_keys = st.secrets.get("user_keys", {})
    if email in user_keys:
        return user_keys[email]
    fallback = st.secrets.get("litellm_key")
    if fallback:
        st.sidebar.warning(f"No personal LiteLLM key for {email} -- using the shared fallback key.")
        return fallback
    st.error(f"No LiteLLM key available for {email}. Ask an admin to register one (Section 10.4).")
    st.stop()


MY_LITELLM_KEY = get_user_litellm_key(USER_EMAIL)

# --------------------------------------------------------------------------
# Cached resources -- loaded once per Streamlit session, not on every rerun
# --------------------------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

@st.cache_resource
def get_remote_collection():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_collection(CHROMA_COLLECTION_NAME)

@st.cache_resource
def load_local_indexes():
    # lexical_weights.pkl is a plain list, one entry per chunk, in the same order as
    # chunks.pkl (see notebook 1, Section 4.3) -- it's not keyed by chunk_id. Rebuild the
    # chunk_id -> weights dict here exactly as notebook 1, Section 5.3 does in-memory.
    with open(LEXICAL_WEIGHTS_PATH, "rb") as f:
        sparse_weights = pickle.load(f)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    with open(PAGE_TEXT_PATH, "rb") as f:
        page_full_text_by_url = pickle.load(f)
    sparse_index = {c.chunk_id: w for c, w in zip(chunks, sparse_weights)}
    chunk_lookup = {c.chunk_id: c for c in chunks}
    return sparse_index, chunk_lookup, page_full_text_by_url


# --------------------------------------------------------------------------
# Retrieval + generation -- identical to Section 12, only the LLM call now carries
# MY_LITELLM_KEY (this visitor's own key) instead of a single shared constant.
# --------------------------------------------------------------------------

def dense_search_remote(query_dense_vec, collection, top_k=10):
    result = collection.query(query_embeddings=[query_dense_vec.tolist()], n_results=top_k)
    return list(zip(result["ids"][0], [1 - d for d in result["distances"][0]]))


def sparse_search_local(query_lexical_weights, sparse_index, bge_model, top_k=10):
    scores = [(cid, bge_model.compute_lexical_matching_score(query_lexical_weights, w))
              for cid, w in sparse_index.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def reciprocal_rank_fusion(ranked_lists, k=60):
    rrf_scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(query, bge_model, collection, sparse_index, chunk_lookup,
                     top_k_each=10, top_k_final=5):
    encoded = bge_model.encode([query], return_dense=True, return_sparse=True)
    query_dense = encoded["dense_vecs"][0]
    query_sparse = encoded["lexical_weights"][0]

    dense_results = dense_search_remote(query_dense, collection, top_k=top_k_each)
    sparse_results = sparse_search_local(query_sparse, sparse_index, bge_model, top_k=top_k_each)
    fused = reciprocal_rank_fusion([dense_results, sparse_results], k=60)[:top_k_final]

    dense_sim_lookup = dict(dense_results)
    enriched = []
    for chunk_id, rrf_score in fused:
        chunk = chunk_lookup[chunk_id]
        enriched.append({
            "text": chunk.text,
            "source_url": chunk.source_url,
            "source_title": chunk.source_title,
            "heading_path": getattr(chunk, "heading_path", ""),
            "rrf_score": rrf_score,
            "cosine_similarity": dense_sim_lookup.get(chunk_id),
        })
    return enriched


def build_context_block(results, page_full_text_by_url, expand_top_n=EXPAND_TOP_N_TO_FULL_PAGE):
    blocks = []
    for i, r in enumerate(results, 1):
        label = f"{r['source_title']} — {r['heading_path']}" if r.get("heading_path") else r["source_title"]
        if i <= expand_top_n and r["source_url"] in page_full_text_by_url:
            body = page_full_text_by_url[r["source_url"]][:MAX_EXPANDED_CHARS]
            blocks.append(f"[Source {i}: {label} ({r['source_url']}) -- FULL PAGE]\n{body}")
        else:
            blocks.append(f"[Source {i}: {label} ({r['source_url']})]\n{r['text']}")
    return "\n\n".join(blocks)


def call_remote_llm(messages, model=LITELLM_MODEL_NAME, sampling_options=None):
    """Same OpenAI-compatible call as Section 12, but authenticated with THIS visitor's own key."""
    payload = {"model": model, "messages": messages}
    if sampling_options:
        payload.update(sampling_options)
    response = requests.post(
        f"{LITELLM_PROXY_URL}/v1/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {MY_LITELLM_KEY}"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate_answer_remote(query, results, page_full_text_by_url, history=(), sampling_options=None):
    context = build_context_block(results, page_full_text_by_url)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context above, and cite the source title(s) you used."
    )
    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
    for role, content in history[-(2 * MAX_HISTORY_TURNS):]:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})
    return call_remote_llm(messages, sampling_options=sampling_options)


def generate_answer_llm_only(query, history=(), sampling_options=None):
    messages = [{"role": "system", "content": LLM_ONLY_SYSTEM_PROMPT}]
    for role, content in history[-(2 * MAX_HISTORY_TURNS):]:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})
    return call_remote_llm(messages, sampling_options=sampling_options)


def rag_chat(query, bge_model, collection, sparse_index, chunk_lookup, page_full_text_by_url, history=()):
    '''
    Full pipeline: retrieve, map the best cosine similarity to a confidence band (7.1), then
    generate accordingly. Returns (answer, results, top_similarity, mode) -- `mode` drives the
    badge/sources in the UI.
    '''
    results = hybrid_retrieve(query, bge_model, collection, sparse_index, chunk_lookup)
    cosine_scores = [r["cosine_similarity"] for r in results if r["cosine_similarity"] is not None]
    top_similarity = max(cosine_scores) if cosine_scores else 0.0

    mode, sampling_options = sampling_for_similarity(top_similarity)

    if mode == "llm_only":
        answer = generate_answer_llm_only(query, history, sampling_options)
        answer = (
            "\U0001F50E Nothing in the documentation clearly matched this question, so this "
            "answer comes from the model's general knowledge, **not** your lab's documentation:"
            "\n\n" + answer
        )
        return answer, results, top_similarity, mode

    answer = generate_answer_remote(query, results, page_full_text_by_url, history, sampling_options)
    if mode == "rag_hedged":
        answer = (
            "⚠️ I found some possibly related information, but I'm not fully "
            "confident it answers your exact question. Please double-check against the "
            "sources below.\n\n" + answer
        )
    return answer, results, top_similarity, mode


def confident_sources(results, min_cosine=SOURCE_LINK_MIN_COSINE, max_links=MAX_SOURCE_LINKS):
    '''
    Which sources are worth citing as clickable links: individually high cosine similarity,
    not just "present in the fused top-k" (see the module-level comment above). Deduplicated by
    (title, url) -- a page can contribute several chunks, each keeps only its best cosine for
    ranking -- and capped at `max_links`.
    '''
    best_cosine_by_source = {}
    for r in results:
        cos = r["cosine_similarity"]
        if cos is None or cos <= min_cosine:
            continue
        key = (r["source_title"], r["source_url"])
        if key not in best_cosine_by_source or cos > best_cosine_by_source[key]:
            best_cosine_by_source[key] = cos
    ranked = sorted(best_cosine_by_source.items(), key=lambda kv: kv[1], reverse=True)
    return [key for key, _cos in ranked[:max_links]]


def sources_markdown(results):
    sources = confident_sources(results)
    if not sources:
        return ""
    return "  \n".join(f"\U0001F517 [{title}]({url})" for title, url in sources)


# --------------------------------------------------------------------------
# Themed sidebar (14.2) -- streamlit-option-menu instead of a plain status list
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"### 👋 {USER_NAME}")
    selected = option_menu(
        menu_title=None,
        options=["Chat", "My account"],
        icons=["chat-dots", "person-circle"],
        default_index=0,
    )
    st.divider()
    st.markdown("**🔌 Connection status**")
    try:
        bge_model = load_embedding_model()
        st.success("BGE-M3 loaded locally")
    except Exception as e:
        st.error(f"BGE-M3 failed to load: {e}")
        st.stop()
    try:
        collection = get_remote_collection()
        st.success(f"Vector store connected ({collection.count()} chunks)")
    except Exception as e:
        st.error(f"Cannot reach remote vector store: {e}")
        st.stop()
    try:
        sparse_index, chunk_lookup, page_full_text_by_url = load_local_indexes()
        st.success(f"Sparse index loaded ({len(sparse_index)} entries)")
    except Exception as e:
        st.error(f"Cannot load local sparse index: {e}")
        st.stop()
    st.divider()
    st.markdown("**🎯 Confidence legend**")
    for style in MODE_STYLE.values():
        st.markdown(
            f'<span class="mode-badge" style="background:{style["color"]}22; color:{style["color"]};">'
            f'{style["label"]}</span>',
            unsafe_allow_html=True,
        )
    st.divider()
    if AUTH_MODE == "demo" and "_authenticator" in st.session_state:
        st.session_state["_authenticator"].logout(button_name="Log out", location="sidebar")
    elif AUTH_MODE == "oidc":
        st.button("Log out", on_click=st.logout)

# --------------------------------------------------------------------------
# "My account" page -- confirms the identity/key/history isolation to the visitor
# --------------------------------------------------------------------------

if selected == "My account":
    st.title("My account")
    st.write(f"**Name:** {USER_NAME}")
    st.write(f"**Email:** {USER_EMAIL}")
    st.write(f"**Auth mode:** `{AUTH_MODE}`")
    st.info(
        "Your chat history and your LiteLLM key are both scoped to your own session -- no other "
        "visitor to this app can see either, and your key's budget (Section 10.4) is tracked "
        "under your identity alone."
    )
    st.stop()

# --------------------------------------------------------------------------
# Chat page -- identical UI to Section 12, history keyed defensively by email
# (st.session_state is already per-session; this namespacing is belt-and-braces clarity,
# not a fix for an actual leak -- see 14.0).
# --------------------------------------------------------------------------

st.markdown(
    f'<div class="hero"><h1>🔐 Lab RAG Assistant</h1>'
    f'<p>Signed in as {USER_EMAIL} &nbsp;·&nbsp; Query encoding: local &nbsp;·&nbsp; '
    f'Vector store &amp; LLM: remote</p></div>',
    unsafe_allow_html=True,
)

history_key = f"history::{USER_EMAIL}"
if history_key not in st.session_state:
    st.session_state[history_key] = []  # dicts: role, content, and (for assistant) mode/sources/similarity/results

for entry in st.session_state[history_key]:
    avatar = "🧑‍💻" if entry["role"] == "user" else "🧪"
    with st.chat_message(entry["role"], avatar=avatar):
        if entry["role"] == "assistant":
            mode_badge(entry.get("mode"))
        st.markdown(entry["content"])
        if entry.get("sources"):
            st.markdown(f'<div class="sources-box">{entry["sources"]}</div>', unsafe_allow_html=True)
        if entry.get("top_similarity") is not None:
            st.caption(f"Top cosine similarity: {entry['top_similarity']:.3f}")
        if entry.get("results") and entry.get("mode") != "llm_only":
            with st.expander("🔍 Retrieved chunks (debug)"):
                for r in entry["results"]:
                    sim = f"{r['cosine_similarity']:.3f}" if r["cosine_similarity"] is not None else "n/a"
                    st.markdown(f"**[{r['source_title']}]({r['source_url']})** — cos={sim}, rrf={r['rrf_score']:.4f}")
                    st.caption(r["text"][:300])

user_query = st.chat_input("Ask something about the lab documentation...")

if user_query:
    st.session_state[history_key].append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_query)

    llm_history = [(h["role"], h["content"]) for h in st.session_state[history_key][:-1]]

    with st.chat_message("assistant", avatar="🧪"):
        with st.spinner("Retrieving + generating..."):
            answer, results, top_similarity, mode = rag_chat(
                user_query, bge_model, collection, sparse_index, chunk_lookup, page_full_text_by_url,
                history=llm_history,
            )
        mode_badge(mode)
        st.markdown(answer)
        sources = sources_markdown(results) if mode != "llm_only" else ""
        if sources:
            st.markdown(f'<div class="sources-box">{sources}</div>', unsafe_allow_html=True)
        st.caption(f"Top cosine similarity: {top_similarity:.3f}")
        if mode != "llm_only":
            with st.expander("🔍 Retrieved chunks (debug)"):
                for r in results:
                    sim = f"{r['cosine_similarity']:.3f}" if r["cosine_similarity"] is not None else "n/a"
                    st.markdown(f"**[{r['source_title']}]({r['source_url']})** — cos={sim}, rrf={r['rrf_score']:.4f}")
                    st.caption(r["text"][:300])

    st.session_state[history_key].append({
        "role": "assistant",
        "content": answer,
        "mode": mode,
        "sources": sources,
        "top_similarity": top_similarity,
        "results": results,
    })
