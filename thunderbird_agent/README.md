# LaboBots Mail Agent — Thunderbird extension

Draft email replies — or compose brand-new emails from a short instruction — with a local (Ollama)
or remote (LiteLLM proxy, same workshop server as notebook 2) LLM, inserted directly into
Thunderbird's own compose window — so your real signature and the real Send button are the ones
used, not a copy-paste from somewhere else.

See `03_thunderbird_agent.ipynb` (repo root) for the pedagogical explanation: what an agent is,
why this plugin qualifies as one, its architecture, and why Thunderbird. This directory is the
actual, standalone deliverable — not a notebook, on purpose, since a browser extension can't run
inside a Jupyter kernel.

## Install

### Permanent install (recommended)

1. Build the `.xpi` package (a zip with `manifest.json` at its root):
   ```bash
   ./thunderbird_agent/build.sh      # -> thunderbird_agent/dist/labobots-mail-agent-<version>.xpi
   ```
2. In Thunderbird: **Tools -> Add-ons and Themes** (or the hamburger menu -> *Add-ons and Themes*),
   click the gear icon -> **Install Add-on From File...** and pick the `.xpi` from `dist/`.
   Thunderbird does not require add-ons to be signed, so the unsigned package installs directly
   and survives restarts.
3. The extension icon appears as a single permanent button in the main Thunderbird window toolbar
   (a `browser_action`); clicking it opens the agent in its **own separate window** (not a popup
   that closes when you click away), which stays open independently of Thunderbird -- put it next
   to the main window and keep reading/replying to other emails while a draft generates. Clicking
   the icon again just refocuses that window instead of opening a second one.

To update after editing the sources: bump `version` in `manifest.json`, rebuild, reinstall.

### Temporary install (while developing)

**Tools -> Developer Tools -> Debug Add-ons** (or `about:debugging` -> *This Thunderbird*) ->
**Load Temporary Add-on...** -> select `manifest.json` in this directory. Thunderbird unloads it
on restart; use the **Reload** button there after each code change.

## Configure

Open the extension's **Options** page (right-click the icon → *Manage Extension* → *Preferences*,
or the "Settings" link inside the agent window itself):

- **Local backend**: defaults already match notebook 1 (`http://localhost:11434`, `llama3.2:3b`)
  — only change these if you used different values there.
- **Remote backend**: same values as `rag_workshop/.streamlit/secrets.toml` (notebook 2, Section
  12.3 / 14.1) — proxy URL, model name, and your own participant key. Needs the SSH tunnel open:
  ```bash
  ./rag_workshop/manage_remote_rag.sh tunnel
  ```

**Ollama and the 403 error**: Ollama rejects requests coming from a `moz-extension://` origin.
The extension rewrites that header for `localhost` / `127.0.0.1`, so it should work out of the
box; if you still get HTTP 403, start Ollama with `OLLAMA_ORIGINS="moz-extension://*" ollama serve`
(or, for the systemd service, `sudo systemctl edit ollama` and add
`Environment="OLLAMA_ORIGINS=moz-extension://*"`).

Only `localhost` / `127.0.0.1` URLs are allowed by the manifest's host permissions; to point a
backend elsewhere, add that host to `permissions` in `manifest.json` and rebuild.

Nothing is sent anywhere until you click **Generate draft**; nothing is stored anywhere except
this Thunderbird profile's local extension storage (see the notebook's "memory" section).

## Use

### Reply to an email

1. Click the LaboBots Mail Agent icon in the main toolbar to open the agent window (it stays open
   independently of Thunderbird's own window until you close it).
2. In Thunderbird's main window, open/select an email. The agent window's **Reply** tab follows
   whatever message is currently displayed there and updates automatically. If the email has
   attachments, it lists each one with an icon showing whether its content will be used: 📄
   text/PDF (content included), 🖼️ image (not read), 📎 anything else (not read). No extra step
   needed — readable attachments are folded into the prompt automatically.
3. Pick **Local** or **Remote**, optionally add steering instructions (tone, what to say, length),
   click **Generate draft**. The agent window keeps working while this runs — switch back to
   Thunderbird's main window and read or reply to other emails if you like; just don't expect the
   Reply tab to follow you there until this draft is done, or the "Insert into reply" target would
   change under you (it pauses that tracking automatically while a generation is in flight).
4. Edit the draft inline if you want, then **Insert into reply** — this opens Thunderbird's own
   reply window with your draft as the body, your signature and quoted original still exactly
   where Thunderbird normally puts them. Review, then hit **Send** yourself — the extension never
   sends anything on its own.

### Compose a new email

1. Open the agent window (see above — works even with no message open) and switch to the **New
   email** tab.
2. Optionally fill in **To** and **Subject** — leave Subject blank to let the LLM suggest one from
   your instructions.
3. Describe what the email should say, pick **Local** or **Remote**, click **Generate draft**.
4. Review (and edit) the subject and body, then **Open in compose window** — this opens a real
   Thunderbird compose tab with your draft as the body and your signature already in place.
   Review, then hit **Send** yourself, exactly as with a reply.

## Attachments

`background.js`'s `getAttachmentsWithText()` reads two kinds of attachments directly, so the
draft can reference their actual content, not just their filename:

- **Text-like files** (`.txt`, `.md`, `.csv`, `.log`, `.json`, `.yaml`, or any `text/*` /
  `application/json` MIME type) — read verbatim via the File API, capped at 4000 characters each.
- **PDFs** — text is extracted with a vendored copy of [pdf.js](vendor/pdfjs/README.md) (Mozilla's
  own PDF engine, Apache-2.0, bundled locally so the extension keeps working fully offline and
  never loads code from a remote CDN), capped at 20 pages and 4000 characters.

Anything else (images, `.docx`/`.xlsx`, archives, a scanned/image-only PDF with no text layer) is
only *noticed* — named in the agent window and mentioned to the model by filename — never guessed
at, since there's no reliable way to read it here. No OCR.

## Mailbox knowledge base (local RAG)

Optional, off by default. When enabled in Options, the extension indexes a local IndexedDB store
-- chunked, embedded, and searched locally -- so draft generation (both Reply and New email) can
pull in relevant context from your own past emails, not just the one being answered. By default
this covers every folder of every account; pick one **account** in Options to scope it to just
that mailbox, and once an account is picked, optionally check specific **folders** under it (e.g.
just "Sent" or one project folder) instead of indexing the whole account -- handy for a large
mailbox you don't want to fully scrape just to get useful context. Leaving every folder checkbox
unchecked indexes every folder of that account, same as before. Narrowing the scope (account or
folders) takes effect on search immediately, even for data already indexed outside it; use
**Clear index** + **Pull now** if you also want to remove that old data from disk.

- **Strictly local**: embeddings always go through your local Ollama instance
  (`ollama pull nomic-embed-text` first), *regardless* of whether you picked Local or Remote as the
  drafting backend. Your mailbox content is never sent to the shared LiteLLM proxy for this
  feature -- only the final chat prompt (which may include short excerpts of matched context) goes
  wherever you chose for drafting, exactly as any other steering instruction would.
- **Incremental**: indexed by each email's stable Message-ID, so re-running a pull (or receiving
  new mail) never re-embeds a message twice -- a pull is safe to interrupt and resume.
- **Updates automatically** as new mail arrives (where your Thunderbird version supports it) and
  on a periodic pull (configurable interval, Options) that also covers any folder automatic
  indexing missed; a **Pull now** button in Options also triggers an immediate full pass, and
  **Clear index** wipes it to start over.
- Retrieval silently returns nothing if the match quality is too low -- it never forces irrelevant
  mailbox content into a draft.

## Known limitations (see the notebook for the "why")

- Body extraction from the MIME tree is a simplified walker (`background.js`,
  `extractBodyFromPart`) — good enough for typical plain-text/HTML emails, not a full MIME parser.
- Attachment support is text files + PDFs only (see above) — no OCR, no Office formats.
- No streaming: the agent window's status line just says "Generating..." until the full draft
  comes back (can be slow on a CPU-only local model for a long email, more so with a multi-page
  PDF attached) — but the agent window no longer blocks you from using the rest of Thunderbird
  while it waits, since it's a separate window rather than a popup.
- History used for style context is a flat local list, not a real embedding-based memory — see
  the notebook's discussion of why that's a deliberate scope choice for this workshop.
