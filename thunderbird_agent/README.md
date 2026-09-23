# LaboBots Mail Agent — Thunderbird extension

Draft email replies with a local (Ollama) or remote (LiteLLM proxy, same workshop server as
notebook 2) LLM, inserted directly into Thunderbird's own reply window — so your real signature
and the real Send button are the ones used, not a copy-paste from somewhere else.

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
3. The extension icon appears in the message header toolbar when you're viewing a message (it's
   a `message_display_action`, not a permanent toolbar button -- see the notebook for why).

To update after editing the sources: bump `version` in `manifest.json`, rebuild, reinstall.

### Temporary install (while developing)

**Tools -> Developer Tools -> Debug Add-ons** (or `about:debugging` -> *This Thunderbird*) ->
**Load Temporary Add-on...** -> select `manifest.json` in this directory. Thunderbird unloads it
on restart; use the **Reload** button there after each code change.

## Configure

Open the extension's **Options** page (right-click the icon → *Manage Extension* → *Preferences*,
or the "Settings" link inside the popup itself):

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

1. Open an email.
2. Click the LaboBots Mail Agent icon. If the email has attachments, the popup lists each one
   with an icon showing whether its content will be used: 📄 text/PDF (content included), 🖼️
   image (not read), 📎 anything else (not read). No extra step needed — readable attachments are
   folded into the prompt automatically.
3. Pick **Local** or **Remote**, optionally add steering instructions (tone, what to say, length),
   click **Generate draft**.
4. Edit the draft inline if you want, then **Insert into reply** — this opens Thunderbird's own
   reply window with your draft as the body, your signature and quoted original still exactly
   where Thunderbird normally puts them. Review, then hit **Send** yourself — the extension never
   sends anything on its own.

## Attachments

`background.js`'s `getAttachmentsWithText()` reads two kinds of attachments directly, so the
draft can reference their actual content, not just their filename:

- **Text-like files** (`.txt`, `.md`, `.csv`, `.log`, `.json`, `.yaml`, or any `text/*` /
  `application/json` MIME type) — read verbatim via the File API, capped at 4000 characters each.
- **PDFs** — text is extracted with a vendored copy of [pdf.js](vendor/pdfjs/README.md) (Mozilla's
  own PDF engine, Apache-2.0, bundled locally so the extension keeps working fully offline and
  never loads code from a remote CDN), capped at 20 pages and 4000 characters.

Anything else (images, `.docx`/`.xlsx`, archives, a scanned/image-only PDF with no text layer) is
only *noticed* — named in the popup and mentioned to the model by filename — never guessed at,
since there's no reliable way to read it here. No OCR.

## Known limitations (see the notebook for the "why")

- Body extraction from the MIME tree is a simplified walker (`background.js`,
  `extractBodyFromPart`) — good enough for typical plain-text/HTML emails, not a full MIME parser.
- Attachment support is text files + PDFs only (see above) — no OCR, no Office formats.
- The popup closes if you click elsewhere in Thunderbird; a draft being generated is then lost.
- No streaming: the popup blocks until the full draft is generated (can be slow on a CPU-only
  local model for a long email, more so with a multi-page PDF attached).
- History used for style context is a flat local list, not a real embedding-based memory — see
  the notebook's discussion of why that's a deliberate scope choice for this workshop.
