# LaboBots RAG workshop

Hands-on material for **LaboBots**, a training school organized by IJCLab in partnership with the
CNRS AISSAI Center's *AI4Metascience* thematic quarter, held **29 September – 2 October 2026** at
Domaine Saint-Paul, Saint-Rémy-lès-Chevreuse. [Event page](https://indico.ijclab.in2p3.fr/event/13661/)

The school trains system administrators and developers from French public research laboratories to
size a local server for a conversational-agent workload and build a working RAG (Retrieval-Augmented
Generation) service around it — from first principles up to a secured, multi-user deployment.
Prerequisite: solid Python.

Three notebooks carry the actual teaching content; everything else in this repo is the operational
tooling that supports running the workshop (server setup, key provisioning, corpus rebuilds):

1. **`01_hybrid_RAG_from_scratch_sections1-7.ipynb`** — build a hybrid (dense + sparse) RAG pipeline
   entirely on one laptop: scraping, chunking, BGE-M3 embeddings, ChromaDB, Reciprocal Rank Fusion,
   and a confidence-driven generation policy with a local Ollama model. Section 7.6 (bonus) adds
   three LLM-summary-based retrieval strategies and measures all seven side by side — see
   "Summarization cache" below.
2. **`02_distributed_architecture_streamlit_litellm.ipynb`** — split that same pipeline across
   machines (a remote vector store, a remote LLM behind a LiteLLM proxy with per-participant keys)
   and expose it through a Streamlit chat client, with and without authentication.
3. **`03_thunderbird_agent.ipynb`** — what an *agent* is (vs. a chatbot), and how `thunderbird_agent/`
   (a real, standalone Thunderbird extension — not a notebook, see below) turns notebooks 1-2's
   chat pipeline into one: reads the email you're viewing, drafts a reply with the local or remote
   LLM, and inserts it into Thunderbird's own reply window — real signature, real Send button.

This directory is the single project root and the GitHub repository root. There is no nested project
for the environment notebook.

## Repository layout

```text
LaboBots_RAG/
├── 01_hybrid_RAG_from_scratch_sections1-7.ipynb   # single-laptop RAG, from scratch
├── 02_distributed_architecture_streamlit_litellm.ipynb  # split across machines + Streamlit
├── 03_thunderbird_agent.ipynb   # what an agent is + walkthrough of thunderbird_agent/ below
├── 00_python_environments_and_uv_project_setup.ipynb  # uv-focused env tutorial; Section 0 inits notebooks 1-2's real environment
├── pyproject.toml
├── uv.lock
├── thunderbird_agent/            # standalone Thunderbird extension -- not a notebook, see 03
│	├── manifest.json
│	├── background.js              # the only thing that calls the LLM or Thunderbird's compose API
│	├── popup/                     # backend choice + steering prompt UI
│	├── options/                   # local/remote backend config (mirrors secrets.toml)
│	├── icons/
│	├── build.sh                   # packages the extension as dist/*.xpi (gitignored)
│	└── README.md                  # install/usage instructions for this extension specifically
├── assets/
│	└── diagrams/                  # architecture SVGs embedded in notebooks 1-2 (LLM, ChromaDB, dense/sparse vectors)
└── rag_workshop/
	├── corpus/                    # scraped pages + generated indexes; all gitignored except .gitkeep-level structure
	├── chunk_types.py              # shared Chunk dataclass (see "Why chunk_types.py exists" below)
	├── rebuild_corpus.py           # full-site crawl -> chunk -> embed -> push to remote Chroma
	├── summarize_corpus.py         # standalone, resumable notebook 1 Section 7.6 (LLM summaries)
	├── streamlit_app.py            # single shared key, simple chat client
	├── streamlit_app_secure.py     # per-user auth (OIDC or demo) + per-user LiteLLM key
	├── create_demo_accounts.py     # one personalized secrets.toml per participant (demo auth)
	├── manage_litellm.sh           # remote Ollama + LiteLLM proxy + Postgres + participant keys
	├── manage_remote_rag.sh        # remote Chroma lifecycle + SSH tunnel helper (see note below)
	├── verify_remote_chroma.py     # local-vs-remote Chroma comparison, used by manage_remote_rag.sh verify
	├── setup_uv.sh                 # one-shot local environment setup
	└── .streamlit/
		├── config.toml          # theme colors, safe to commit
		└── secrets.toml         # personal keys/tunnel config, gitignored, never commit
```

Local virtual environments, Chroma persistence, embeddings, pickled indexes, scraped corpus dumps,
the summarization cache, and secrets are ignored by Git (see `.gitignore`) — the whole of
`rag_workshop/corpus/` is generated or copied during workshop setup, not
checked in.

**Why `chunk_types.py` exists**: `Chunk` used to be defined inline in a notebook 1 cell. Pickling it
from there records the class under `__main__` (the *kernel's* module), which a separate process (the
Streamlit apps unpickling `chunks.pkl`) cannot resolve. Moving it into a real, importable module fixes
that for good — both the notebook and the Streamlit apps import from `chunk_types.py`.

## Local setup

Requires Python 3.11 to 3.13; `uv` is the recommended tool. From the workspace root:

```bash
./rag_workshop/setup_uv.sh          # one-shot setup: venv, dependencies, Jupyter kernel
uv sync --extra embeddings          # BGE-M3 (notebook 1, rebuild_corpus.py, both streamlit apps)
uv sync --extra secure-app          # streamlit_app_secure.py
```

Select the kernel `Python (LaboBots RAG workshop)` in VS Code and run project commands with `uv run`.
The `embeddings` extra installs BGE-M3's dependencies (including `torchvision`, needed only because
`transformers` lazy-imports some unused vision submodules that reference it -- not used for anything
in this workshop). The `secure-app` extra installs what `streamlit_app_secure.py` needs (`Authlib`,
`streamlit-authenticator`, `streamlit-option-menu`).

## Building the corpus

Notebook 1, Section 2, controls how many pages get crawled from `doc.cc.in2p3.fr` — pick one, comment
out the other:

```python
MAX_PAGES = 15          # quick local iteration while working through the notebook (~15s)
# MAX_PAGES = None        # full site, no cap (~110-150 pages) -- use this for the real demo
```

The crawler follows the site's own navigation links (plus a sitemap, if one exists) and skips binary
downloads (`.zip`, `.tar.gz`, `.pdf`, images, ...) so it never tries to parse an archive as HTML.

For the actual workshop deployment, `rag_workshop/rebuild_corpus.py` runs the same pipeline
end-to-end (crawl -> chunk -> embed with BGE-M3 -> push to the remote Chroma collection) with
`MAX_PAGES = None` by default:

```bash
# with the SSH tunnel to Machine A open (see below), from the repo root:
python3 rag_workshop/rebuild_corpus.py
```

This takes several minutes (a polite ~1 request/second crawl, then CPU-bound BGE-M3 encoding) --
but only the first time. Each of the three expensive stages (crawl, chunk, embed) caches its
output to `rag_workshop/corpus/` and is skipped on the next run if that output is still there,
valid, and matches the current corpus. So if the run fails at the last step (e.g. the tunnel
dropped right before the Chroma push), re-running the script does not redo the crawl or the
BGE-M3 encoding -- it goes straight back to pushing to Chroma. Set `FORCE_RECRAWL` /
`FORCE_RECHUNK` / `FORCE_REEMBED` at the top of the script to force a stage to redo its work
regardless of what's cached.

## Summarization cache (Notebook 1, Section 7.6, bonus)

Section 7.6 compares seven retrieval strategies, three of which need an LLM-generated summary and
a keyword string for every chunk — the slowest thing in either notebook (roughly 20-45s per chunk
on CPU, so hours on a full-site corpus). Both the notebook cell and the standalone script below
read and write the same cache, `rag_workshop/corpus/chunk_summaries.json` (human-readable: chunk
id, page title, URL, raw text, summary, keywords) and `chunk_summaries_embeddings.npz` (their
embeddings) — whichever one fills the cache, the other picks it up with nothing recomputed.

For anything beyond the quick demo corpus, prefer the standalone script over running it inline in
the notebook:

```bash
python3 rag_workshop/summarize_corpus.py
```

It checkpoints every 20 newly-computed chunks (so `Ctrl+C` and a later resume loses at most that
much), and periodically unloads the Ollama model (`keep_alive=0` every 50 chunks) so its memory
footprint doesn't keep climbing across hundreds of sequential calls until the machine starts
swapping — left running inline in a notebook kernel across a full corpus, we saw Ollama alone grow
to 7+ GB RSS and push the whole machine into swap. `CHECKPOINT_EVERY` / `UNLOAD_MODEL_EVERY` /
`SUMMARY_MAX_CHUNKS` are constants at the top of the script.

Section 7.6.5 can then push the cached summaries into their own Chroma collection
(`ccin2p3_docs_summaries`) — it only ever reads the cache, never triggers summarization itself.

## Remote vector store (Chroma)

`rag_workshop/manage_remote_rag.sh` manages the remote Chroma process and the local SSH tunnel:

```bash
./rag_workshop/manage_remote_rag.sh tunnel   # opens BOTH the Chroma (8000) and LiteLLM (4000)
                                              # forwards in one SSH connection -- prefer this over
                                              # separate manual `ssh -L ...` invocations
./rag_workshop/manage_remote_rag.sh status   # check the tunnel + remote Chroma process
./rag_workshop/manage_remote_rag.sh stop-tunnel
```

**Populating the collection**: the currently-used method is `rebuild_corpus.py` inserting directly
into the live remote Chroma server over HTTP (`chromadb.HttpClient`) — safe to run against a Chroma
server that's already running, no file copy or restart involved. `manage_remote_rag.sh prepare` /
`copy` document an older alternative (build `rag_workshop/chroma_db/` locally via notebook 1, then
rsync that directory to the remote host and restart `chroma run` there); it still works, but isn't
the path currently used, and `verify_remote_chroma.py` (which compares a local Chroma directory
against the remote one) assumes that older workflow — it will report a mismatch against a corpus
built directly via `rebuild_corpus.py`, since there's no matching local copy to compare against.

## LiteLLM proxy, Postgres, and participant keys

```bash
./rag_workshop/manage_litellm.sh install-db   # once: installs Postgres + creates the litellm DB/role.
                                               # Needs an account with sudo on the remote host --
                                               # doesn't have to be the same account LiteLLM runs as:
                                               #   ADMIN_USER=youradminaccount manage_litellm.sh install-db
./rag_workshop/manage_litellm.sh start        # prompts for a master key + the DB password from install-db
./rag_workshop/manage_litellm.sh status
./rag_workshop/manage_litellm.sh create-keys --duration 20d --budget 30
```

`install-db` is required before the first `create-keys` — LiteLLM's virtual-key management needs a
Postgres-backed database; without it, `start` still runs (in-memory mode) but `/key/generate` fails.

The master key and DB password are entered interactively and never printed. Participant keys are
generated remotely into `$HOME/rag_workshop/participant-keys.tsv` (mode 600); copy it down with:

```bash
scp -P 22003 labobots@195.221.220.18:rag_workshop/participant-keys.tsv .
```

**Keys default to 8h** (`--duration 8h`), overridable per call, e.g. `--duration 20d` for a
multi-day event so you don't have to regenerate every morning — lower is safer if a key might leak
(screen share, copy-paste in chat), higher means less day-to-day re-running. `--budget` (default
`5`, in USD) is independent of duration — raise it too for a longer-lived key, or it may run out
well before the key itself expires. `create-keys` is idempotent: it deletes any existing key
sharing the same alias before creating the new one, so re-running with the same
`--count`/`--prefix` just refreshes the same `participant-01`..`participant-N` aliases instead of
failing on "alias already exists" — but note this **invalidates every previously issued key for
that alias**, including any already copied into a `secrets.toml` or the Thunderbird extension's
Options page; re-distribute the refreshed `participant-keys.tsv` after every re-run.

**What gets logged**: LiteLLM records every request's full prompt and response in Postgres by
default, tied to the calling key (`LiteLLM_SpendLogs` table) — useful for reviewing the workshop
afterward, but tell participants before the session (see notebook 2, Section 10.7 for the exact
wording and how to disable it if you'd rather not keep this data).

## Participant login accounts (`streamlit_app_secure.py`, demo mode)

`streamlit_app_secure.py`'s `auth_mode = "demo"` needs a username/password per participant
(notebook 2, Section 14.3). Since each participant runs Streamlit on their **own** laptop, their
`secrets.toml` only ever needs to contain **their own** credentials — never the whole group's, the
way the notebook's illustrative alice/bob example does. `rag_workshop/create_demo_accounts.py`
generates that: one personalized, ready-to-use `secrets.toml` per participant, reusing the LiteLLM
key they were already issued (`participant-keys.tsv`, see above) and a freshly generated,
bcrypt-hashed demo password.

```bash
python3 rag_workshop/create_demo_accounts.py
```

Reads `participant-keys.tsv` (repo root) and writes, per participant, into
`rag_workshop/participant-secrets/` (gitignored):

- `<alias>.toml` — a complete `secrets.toml`, ready to copy to `rag_workshop/.streamlit/secrets.toml`
  on that participant's own machine.
- `_distribution-list.tsv` — the plaintext passwords (needed once, to hand out) paired with each
  alias. Hand out **one row at a time**, never the whole file, and delete it once everyone has
  their credentials — it's the only place a password exists in clear.

Each participant's procedure: copy their `<alias>.toml` to `rag_workshop/.streamlit/secrets.toml`,
then `streamlit run rag_workshop/streamlit_app_secure.py` and log in with their alias (e.g.
`participant-07`) and the password from their row of the distribution list.

## Streamlit apps

- `streamlit_app.py`: one shared participant key per person, no login. Confidence-banded answers
  (near-extractive / light synthesis / hedged / general-knowledge-fallback, see notebook 1 Section
  7.1) with clickable sources (only chunks with cosine similarity > 0.6, capped at 10) and a
  configurable amount of conversational memory (`MAX_HISTORY_TURNS`).
- `streamlit_app_secure.py`: same retrieval/generation pipeline, plus an auth gate (OIDC or a
  workshop-only demo login) and a per-user key lookup instead of one shared key.

Both read `rag_workshop/.streamlit/secrets.toml` (gitignored) for connection details and keys — see
notebook 2, Sections 12 and 14, for how to fill it in.

**Tip — view the app inside VS Code instead of a separate browser window**: after
`streamlit run ...` prints its local URL (`http://localhost:8501`), open the command palette
(`Ctrl+Shift+P`) and run **`Simple Browser: Show`**, then paste that URL. The app opens in a VS
Code tab next to your notebook/terminal, so you can keep the Streamlit logs, the code, and the
running app all in one window.

## Thunderbird extension

`thunderbird_agent/` is packaged and installed independently of the Python environment:

```bash
./thunderbird_agent/build.sh        # -> thunderbird_agent/dist/labobots-mail-agent-<version>.xpi
```

Then in Thunderbird: **Add-ons and Themes** -> gear icon -> **Install Add-on From File...**. Full
install, configuration (including Ollama's `OLLAMA_ORIGINS` note) and usage are in
[`thunderbird_agent/README.md`](thunderbird_agent/README.md); the design is explained in
notebook 3.

## Verification

```bash
./rag_workshop/manage_remote_rag.sh verify
./rag_workshop/manage_remote_rag.sh verify --all-ids
```

Only meaningful if the remote corpus was built via the local-build-then-copy path (see above); after
a `rebuild_corpus.py` run, check the collection directly instead (`collection.count()` from a
notebook cell, or the Streamlit sidebar's "Connected to remote vector store (N chunks)" line).

## Contributing

- Keep changes focused and preserve the existing notebook and script structure. This directory is
  the single project root: don't create a nested project for a notebook.
- Serialized `Chunk` objects must come from `rag_workshop/chunk_types.py` (see above); never
  redefine the class in a notebook or script.
- Keep retrieval behavior aligned between notebook 1 and both Streamlit clients.
- The Streamlit apps read their configuration from `rag_workshop/.streamlit/secrets.toml`; don't
  hardcode deployment values (hosts, ports, keys) in the code.
- `rag_workshop/corpus/` and `rag_workshop/chroma_db/` are generated state: rebuild them with the
  scripts, never edit them by hand, and preserve the cache semantics of `rebuild_corpus.py`.
- Prefer the standard library or already-declared dependencies over adding new packages.
- Use ASCII in source code unless existing content requires otherwise.
- There is no formal test suite: check a change by running the narrowest relevant script or app
  startup, then review the diff with `git diff --check`. Test the Thunderbird extension through
  its own install flow.

## Do not commit

`.streamlit/secrets.toml`, any API/participant keys, `participant-keys.tsv`,
`rag_workshop/participant-secrets/`, `.venv/`, and everything under `rag_workshop/corpus/` and
`rag_workshop/chroma_db/` — all already covered by `.gitignore`. Treat `participant-keys.tsv` and
`participant-secrets/` as sensitive even though they're ignored, and never print master keys or
database passwords in commands, logs, or error messages.
