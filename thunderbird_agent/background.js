/*
 * LaboBots Mail Agent -- background script.
 *
 * Owns everything the popup itself shouldn't: reading the displayed message, calling the LLM
 * (local Ollama or remote LiteLLM proxy -- same two backends as notebooks 1 and 2), keeping a
 * small local "writing style" history, and inserting the generated draft into a REAL Thunderbird
 * compose window -- a reply (via compose.beginReply) or a brand-new email (via compose.beginNew)
 * -- so the user's signature and Send button are the native ones -- this extension never sends
 * anything itself.
 *
 * All state lives in browser.storage.local (this profile only, never synced/uploaded anywhere
 * except the two LLM endpoints the user configured in Options).
 */

const DEFAULT_SETTINGS = {
  backend: "local",
  ollama_url: "http://localhost:11434",
  ollama_model: "llama3.2:3b",
  litellm_url: "http://127.0.0.1:4000",
  litellm_model: "workshop-llm",
  litellm_key: "",
  history: [],
  newEmailHistory: [],
  ragEnabled: false,
  ragEmbedModel: "nomic-embed-text",
  ragPullIntervalMinutes: 60,
  ragAccountId: "", // "" = all accounts; set to one account's id to scope indexing/search to it
  ragFolderIds: [], // [] = every folder of the scoped account(s); otherwise only these folder ids
};

const MAX_HISTORY = 20;
const MAX_HISTORY_IN_PROMPT = 3; // how many past drafts get folded into the prompt as style context

const DRAFT_SYSTEM_PROMPT = `You are an email-drafting assistant. You will be shown an email and
asked to draft a reply to it. Rules:
- Reply in the same language as the original email.
- Write the body of the reply. The
  user's own Thunderbird signature is added automatically after your text; do not duplicate it.
- Follow the user's own steering instructions if given (tone, what to say, what to avoid, length).
- Stay concise and professional unless the instructions say otherwise.
- If reference is made to "previous replies" below, they are only style examples (how this
  person usually writes) -- never repeat their content, just match the tone/register.`;

const NEW_EMAIL_SYSTEM_PROMPT = `You are an email-drafting assistant. You will be asked to compose
a brand-new email from scratch -- not a reply to anything. Rules:
- Write in the language the user's instructions are given in, unless told otherwise.
- Write only the body of the email. The user's own Thunderbird signature is added automatically
  after your text; do not duplicate it or invent one.
- Follow the user's instructions for the recipient, purpose, tone, and content.
- If no subject was given, suggest one as the very first line, exactly in the form
  "Subject: ...", followed by a blank line and then the body. If a subject was already given,
  skip straight to the body -- no "Subject:" line.
- Stay concise and professional unless the instructions say otherwise.
- If reference is made to "previous emails" below, they are only style examples (how this person
  usually writes) -- never repeat their content, just match the tone/register.`;

async function getSettings() {
  const stored = await browser.storage.local.get(DEFAULT_SETTINGS);
  return { ...DEFAULT_SETTINGS, ...stored };
}

async function saveSetting(partial) {
  await browser.storage.local.set(partial);
}

/**
 * Thunderbird's messages.getFull() returns a MIME part tree, not a flat body string. This walks
 * it looking for a text/plain part first, falling back to text/html (stripped of tags). It's a
 * simplified extractor for workshop purposes -- a production add-on would use a real MIME
 * library for edge cases (nested multipart/alternative inside multipart/mixed, inline images...).
 */
function extractBodyFromPart(part, preferred = "text/plain") {
  if (!part) return null;
  if (part.contentType && part.contentType.startsWith(preferred) && part.body) {
    return part.body;
  }
  if (part.parts) {
    for (const child of part.parts) {
      const found = extractBodyFromPart(child, preferred);
      if (found) return found;
    }
  }
  return null;
}

function htmlToPlainText(html) {
  const doc = new DOMParser().parseFromString(html, "text/html");
  return doc.body ? doc.body.textContent.trim() : html;
}

/**
 * Pulls every Message-ID this message is threaded with (its own id, plus everything in its
 * References/In-Reply-To headers), so rag.js can later tell whether two messages belong to the
 * same conversation -- a much stronger relevance signal than topical similarity alone. Those two
 * headers aren't exposed on the MessageHeader object the rest of this file works with, so this
 * reads them from the raw MIME headers via getFull() instead.
 */
function extractHeaderIds(values) {
  if (!values) return [];
  const joined = Array.isArray(values) ? values.join(" ") : String(values);
  return (joined.match(/<[^>]+>/g) || []).map((m) => m.slice(1, -1));
}

async function getThreadIds(messageId, headerMessageId) {
  try {
    const full = await browser.messages.getFull(messageId);
    const headers = full.headers || {};
    const refs = extractHeaderIds(headers["references"]);
    const inReplyTo = extractHeaderIds(headers["in-reply-to"]);
    return Array.from(new Set([headerMessageId, ...refs, ...inReplyTo].filter(Boolean)));
  } catch (err) {
    console.error("LaboBots: could not read thread headers for message", messageId, err);
    return [headerMessageId].filter(Boolean);
  }
}

async function getMessageBody(messageId) {
  // Thunderbird 128+ has a dedicated API that already decodes the inline text parts; older
  // versions fall back to walking the MIME tree ourselves.
  if (browser.messages.listInlineTextParts) {
    const parts = await browser.messages.listInlineTextParts(messageId);
    const plain = parts.find((p) => p.contentType === "text/plain");
    if (plain && plain.content.trim()) return plain.content;
    const html = parts.find((p) => p.contentType === "text/html");
    if (html) return htmlToPlainText(html.content);
  }
  const full = await browser.messages.getFull(messageId);
  const plain = extractBodyFromPart(full, "text/plain");
  if (plain) return plain;
  const html = extractBodyFromPart(full, "text/html");
  return html ? htmlToPlainText(html) : "(could not extract a readable body)";
}

// --------------------------------------------------------------------------
// Attachments: text files are read directly; PDFs are parsed with a vendored copy of pdf.js
// (see vendor/pdfjs/README.md). Anything else (images, office docs, archives...) is only
// *noticed* -- named in the prompt, never guessed at -- since we have no reliable way to read it.
// --------------------------------------------------------------------------

const ATTACHMENT_TEXT_MAX_CHARS = 4000; // per attachment, so one huge file can't eat the whole prompt
const ATTACHMENT_PDF_MAX_PAGES = 20; // keeps a long report from stalling a CPU-only local model

const TEXT_ATTACHMENT_EXTENSIONS = [".txt", ".md", ".markdown", ".csv", ".log", ".json", ".yaml", ".yml"];

function isTextAttachment(name, contentType) {
  if (contentType && contentType.startsWith("text/")) return true;
  if (contentType === "application/json") return true;
  const lower = (name || "").toLowerCase();
  return TEXT_ATTACHMENT_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function isPdfAttachment(name, contentType) {
  if (contentType === "application/pdf") return true;
  return (name || "").toLowerCase().endsWith(".pdf");
}

// Loaded lazily (only when an email actually has a PDF attachment) and cached across calls --
// pdf.js is a ~1.8 MB vendored dependency, no reason to pay that cost on every popup open.
let pdfjsLibPromise = null;
function loadPdfJs() {
  if (!pdfjsLibPromise) {
    pdfjsLibPromise = import(browser.runtime.getURL("vendor/pdfjs/pdf.min.mjs")).then((lib) => {
      lib.GlobalWorkerOptions.workerSrc = browser.runtime.getURL("vendor/pdfjs/pdf.worker.min.mjs");
      return lib;
    });
  }
  return pdfjsLibPromise;
}

async function extractPdfText(bytes) {
  const pdfjsLib = await loadPdfJs();
  const doc = await pdfjsLib.getDocument({ data: bytes, isEvalSupported: false }).promise;
  const pageCount = Math.min(doc.numPages, ATTACHMENT_PDF_MAX_PAGES);
  const pageTexts = [];
  for (let pageNum = 1; pageNum <= pageCount; pageNum++) {
    const page = await doc.getPage(pageNum);
    const content = await page.getTextContent();
    pageTexts.push(content.items.map((item) => item.str).join(" "));
  }
  let text = pageTexts.join("\n\n").trim();
  if (doc.numPages > pageCount) {
    text += `\n\n[... ${doc.numPages - pageCount} more page(s) truncated ...]`;
  }
  return text;
}

/**
 * Reads every attachment on a message and returns one summary entry each: `textIncluded` says
 * whether its content made it into `text` (and therefore into the LLM prompt later); `note`
 * explains why not, for attachments we only notice by name (images, office docs, a scanned PDF
 * with no extractable text layer, a read error...). Never throws -- one bad attachment shouldn't
 * block drafting a reply about the rest of the email.
 */
async function getAttachmentsWithText(messageId) {
  const list = await browser.messages.listAttachments(messageId);
  const results = [];
  for (const att of list) {
    const info = { name: att.name, contentType: att.contentType || "", textIncluded: false, text: "", note: "" };
    try {
      if (isTextAttachment(att.name, att.contentType)) {
        const file = await browser.messages.getAttachmentFile(messageId, att.partName);
        info.text = (await file.text()).slice(0, ATTACHMENT_TEXT_MAX_CHARS);
        info.textIncluded = info.text.trim().length > 0;
      } else if (isPdfAttachment(att.name, att.contentType)) {
        const file = await browser.messages.getAttachmentFile(messageId, att.partName);
        const bytes = new Uint8Array(await file.arrayBuffer());
        info.text = (await extractPdfText(bytes)).slice(0, ATTACHMENT_TEXT_MAX_CHARS);
        info.textIncluded = info.text.trim().length > 0;
        if (!info.textIncluded) info.note = "no extractable text (likely a scanned/image-only PDF)";
      } else {
        info.note = "not a text file or PDF -- not read";
      }
    } catch (err) {
      info.note = `could not read this attachment (${err.message})`;
    }
    results.push(info);
  }
  return results;
}

function buildAttachmentsContext(attachments) {
  const all = attachments || [];
  const withText = all.filter((a) => a.textIncluded && a.text.trim());
  const unreadable = all.filter((a) => !(a.textIncluded && a.text.trim()));

  let context = "";
  if (withText.length > 0) {
    const blocks = withText.map((a) => `--- Attachment: ${a.name} ---\n${a.text}`).join("\n\n");
    context += `\n\nThe email has the following attachment(s); their content is included below -- use it as additional context for the reply if relevant:\n\n${blocks}`;
  }
  if (unreadable.length > 0) {
    // Not read (see getAttachmentsWithText's `note`), but still worth the model knowing they
    // exist -- e.g. so a reply can say "I see you attached the invoice" without inventing content
    // for it. Named only, matching the popup's own 🖼️/📎 icons -- see README's Attachments section.
    const names = unreadable.map((a) => `${a.name} (${a.note || "not read"})`).join(", ");
    context += `\n\nThe email also has these attachment(s), which could not be read: ${names}. `
      + "You may acknowledge them by name, but do not invent or guess their content.";
  }
  return context;
}

async function getDisplayedEmail(tabId) {
  // The popup passes the id of the tab it was opened from: the background page has no "current
  // tab" of its own, and getDisplayedMessage() needs to know which tab's message to read.
  const message = await browser.messageDisplay.getDisplayedMessage(tabId);
  if (!message) {
    throw new Error("No single message is currently displayed. Open (or select) one email first.");
  }
  const body = await getMessageBody(message.id);
  const attachments = await getAttachmentsWithText(message.id);
  const threadIds = await getThreadIds(message.id, message.headerMessageId);
  return {
    messageId: message.id,
    headerMessageId: message.headerMessageId,
    subject: message.subject,
    from: message.author,
    body: body.slice(0, 6000), // keep the prompt a reasonable size for a small local model
    attachments,
    threadIds,
  };
}

function buildHistoryContext(history, label = "reply") {
  const recent = history.slice(-MAX_HISTORY_IN_PROMPT);
  if (recent.length === 0) return "";
  const examples = recent
    .map((h, i) => `Previous ${label} example ${i + 1} (style reference only):\n${h.draft}`)
    .join("\n\n");
  return `\n\nHere are a few of this user's own previous ${label}s, for style reference only:\n\n${examples}`;
}

/**
 * fetch() only says "NetworkError" when the server is down or the host isn't covered by the
 * manifest's host permissions (localhost / 127.0.0.1 only); turn that into an actionable message.
 */
async function fetchOrExplain(baseUrl, hint, url, init) {
  try {
    return await fetch(url, init);
  } catch (err) {
    throw new Error(`${hint} (${baseUrl}: ${err.message})`);
  }
}

// `signal` (optional, default undefined) lets the streaming-path fallback abort an in-flight
// non-streaming call; the one-shot handleMessage path passes nothing and behaves exactly as before.
async function callOllama(settings, messages, signal) {
  const resp = await fetchOrExplain(settings.ollama_url, "Cannot reach Ollama -- is 'ollama serve' running?", `${settings.ollama_url}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: settings.ollama_model, messages, stream: false }),
    signal,
  });
  if (resp.status === 403) {
    throw new Error("Ollama refused the request (HTTP 403, origin check). Restart Ollama with OLLAMA_ORIGINS=moz-extension://* -- see the extension README.");
  }
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`Ollama returned HTTP ${resp.status}. ${text} -- is 'ollama serve' running and the model '${settings.ollama_model}' pulled?`);
  }
  const data = await resp.json();
  return data.message.content;
}

async function callLiteLLM(settings, messages, signal) {
  if (!settings.litellm_key) {
    throw new Error("No LiteLLM key configured -- set it in this extension's Options page.");
  }
  const resp = await fetchOrExplain(settings.litellm_url, "Cannot reach the LiteLLM proxy -- is the SSH tunnel open (manage_remote_rag.sh tunnel)?", `${settings.litellm_url}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${settings.litellm_key}`,
    },
    body: JSON.stringify({ model: settings.litellm_model, messages }),
    signal,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`LiteLLM proxy returned HTTP ${resp.status}. ${text} -- is the SSH tunnel to the remote host open (manage_remote_rag.sh tunnel)?`);
  }
  const data = await resp.json();
  return data.choices[0].message.content;
}

/*
 * Streaming variants of the two calls above (same endpoints, same guardrails, stream:true): the
 * answer arrives as a byte stream which is decoded (TextDecoder with stream:true, so a UTF-8
 * multi-octet character cut by a chunk boundary still reassembles) and split into complete
 * lines by StreamLineBuffer before any line is parsed. onToken(token) is invoked for every
 * non-empty token; the complete draft is returned so the caller can post-process it (Subject:
 * extraction...). Errors thrown are the same family as the non-streaming versions plus
 * AbortError when the caller's signal fires -- runStreamed decides what each means.
 */

/**
 * The decode -> line-split -> parse loop, shared by streamOllama and streamLiteLLM (it used to
 * be duplicated verbatim in both). `parse` is an object:
 *   parse.parseLine(line) -> { token?, dataLine? }   (may throw: that aborts the whole stream)
 *   parse.flush()        -> same shape, called ONCE at end-of-stream, only if defined --
 *                            parsers with per-event state (SSEParser) reassemble whatever the
 *                            server left without a closing marker; stateless ones omit it.
 * `noBodyMessage` is the backend-specific wording for the "no readable body" error, which stays
 * per-backend because the two backends have different likely causes.
 * Returns { full, raw, dataLinesSeen }:
 *   full          -- the concatenated non-empty tokens (the draft);
 *   raw           -- the whole decoded body, kept only so a caller can quote a slice of it in
 *                    an error message (assertSseStreamSawData);
 *   dataLinesSeen -- lines parseLine flagged dataLine:true (SSE only; always 0 for Ollama).
 */
async function consumeStream(resp, parse, onToken, noBodyMessage) {
  if (!resp.body) {
    throw new Error(noBodyMessage);
  }
  const decoder = new TextDecoder();
  const buffer = new StreamLineBuffer();
  let full = "";
  let raw = "";
  let dataLinesSeen = 0;
  const emit = (r) => {
    if (r.dataLine) dataLinesSeen += 1;
    if (r.token) {
      full += r.token;
      onToken(r.token);
    }
  };
  const handleLines = (lines) => {
    for (const line of lines) emit(parse.parseLine(line));
  };
  const reader = resp.body.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      // { stream:true } is what tells the decoder to hold back a cut multi-octet character until
      // its continuation bytes arrive with the next chunk.
      const text = decoder.decode(value, { stream: true });
      raw += text;
      handleLines(buffer.feed(text));
    }
    // End of stream: parse whatever partial line the server left without a final newline -- some
    // backends close cleanly instead of sending their own done/[DONE] marker, and that text is
    // real text either way. (If the last line is invalid it throws here -- that is the point of
    // doing it: an unparseable tail is an error, not a silent loss.)
    handleLines([buffer.flush()]);
    // Parsers with per-event state reassemble a trailing event that never received its closing
    // marker; stateless ones (Ollama) have no flush and nothing more to emit.
    if (typeof parse.flush === "function") {
      emit(parse.flush());
    }
  } finally {
    // Release the reader's lock even when we bail out mid-stream (abort, parse error): without
    // this the response body can linger until GC in some engine builds.
    try { reader.releaseLock(); } catch (err) { /* already released */ }
  }
  return { full, raw, dataLinesSeen };
}

async function streamOllama(settings, messages, onToken, signal) {
  const resp = await fetchOrExplain(settings.ollama_url, "Cannot reach Ollama -- is 'ollama serve' running?", `${settings.ollama_url}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: settings.ollama_model, messages, stream: true }),
    signal,
  });
  if (resp.status === 403) {
    throw new Error("Ollama refused the request (HTTP 403, origin check). Restart Ollama with OLLAMA_ORIGINS=moz-extension://* -- see the extension README.");
  }
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`Ollama returned HTTP ${resp.status}. ${text} -- is 'ollama serve' running and the model '${settings.ollama_model}' pulled?`);
  }
  const { full } = await consumeStream(
    resp,
    { parseLine: parseOllamaLine },
    onToken,
    "Ollama returned no readable body for the stream -- this browser build may not support streaming responses."
  );
  return full;
}

async function streamLiteLLM(settings, messages, onToken, signal) {
  if (!settings.litellm_key) {
    throw new Error("No LiteLLM key configured -- set it in this extension's Options page.");
  }
  const resp = await fetchOrExplain(settings.litellm_url, "Cannot reach the LiteLLM proxy -- is the SSH tunnel open (manage_remote_rag.sh tunnel)?", `${settings.litellm_url}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${settings.litellm_key}`,
    },
    body: JSON.stringify({ model: settings.litellm_model, messages, stream: true }),
    signal,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`LiteLLM proxy returned HTTP ${resp.status}. ${text} -- is the SSH tunnel to the remote host open (manage_remote_rag.sh tunnel)?`);
  }
  // ONE parser instance per stream (see SSEParser's doc): it must own the accumulation state of
  // every line of THIS stream, not share it across concurrent generations.
  const sse = new SSEParser();
  const { full, raw, dataLinesSeen } = await consumeStream(
    resp,
    { parseLine: (line) => sse.parseLine(line), flush: () => sse.flush() },
    onToken,
    "The LiteLLM proxy returned no readable body for the stream -- this browser build may not support streaming responses."
  );
  // A 200 body with not a single "data:" line is not an SSE stream (an HTML error page from a
  // broken tunnel, or a one-shot JSON answer from a proxy that ignored stream:true): raising
  // here -- before any token is out, so runStreamed falls back to the non-streaming call --
  // beats resolving with an empty "draft" that Insert would have accepted.
  assertSseStreamSawData(dataLinesSeen, raw);
  return full;
}

/**
 * Builds the exact prompt (system + user message) for a reply to `email` -- RAG context,
 * attachments, style history, all of it. Split out of generateDraft() so the streaming path
 * (runStreamed) reuses the SAME prompt-building code instead of duplicating this block: one
 * source of truth for what the model sees, whether or not the answer streams.
 */
async function buildDraftMessages(email, steeringPrompt) {
  const settings = await getSettings();
  const historyContext = buildHistoryContext(settings.history);
  const attachmentsContext = buildAttachmentsContext(email.attachments);
  const ragChunks = settings.ragEnabled
    ? await Rag.search(email.body, { excludeHeaderMessageId: email.headerMessageId, currentThreadIds: email.threadIds })
    : [];
  // Same search, but restricted to the user's own past emails -- a style reference (how *I*
  // write), kept separate from ragContext's factual mailbox content (how the topic was discussed,
  // by anyone). Same-thread chunks (currentThreadIds) are preferred in both via the thread boost.
  const ownStyleChunks = settings.ragEnabled
    ? await Rag.search(email.body, {
        excludeHeaderMessageId: email.headerMessageId,
        currentThreadIds: email.threadIds,
        requireIsFromMe: true,
        topK: 3,
      })
    : [];
  const ragContext = Rag.buildContextBlock(ragChunks);
  const ownStyleContext = Rag.buildStyleContextBlock(ownStyleChunks);

  const userPrompt = `Original email
From: ${email.from}
Subject: ${email.subject}

${email.body}
${attachmentsContext}${ragContext}${ownStyleContext}
---
${steeringPrompt ? `Steering instructions from the user: ${steeringPrompt}` : "No specific steering instructions -- use your best judgment for a reasonable reply."}${historyContext}

Draft the reply now.`;

  return [
    { role: "system", content: DRAFT_SYSTEM_PROMPT },
    { role: "user", content: userPrompt },
  ];
}

async function generateDraft({ email, steeringPrompt, backend, signal }) {
  const settings = await getSettings();
  const messages = await buildDraftMessages(email, steeringPrompt);
  const draft = backend === "remote"
    ? await callLiteLLM(settings, messages, signal)
    : await callOllama(settings, messages, signal);

  return draft.trim();
}

/**
 * Builds the exact prompt (system + user message) for a brand-new email -- same split-out of
 * prompt building as buildDraftMessages, for the same reason (shared by the streaming path).
 */
async function buildNewEmailMessages(to, subject, steeringPrompt) {
  const settings = await getSettings();
  const historyContext = buildHistoryContext(settings.newEmailHistory || [], "email");
  const query = steeringPrompt || subject;
  const ragChunks = settings.ragEnabled ? await Rag.search(query) : [];
  const ownStyleChunks = settings.ragEnabled ? await Rag.search(query, { requireIsFromMe: true, topK: 3 }) : [];
  const ragContext = Rag.buildContextBlock(ragChunks);
  const ownStyleContext = Rag.buildStyleContextBlock(ownStyleChunks);

  const userPrompt = `New email to compose (not a reply)
${to ? `To: ${to}` : "Recipient: not specified -- write generically."}
${subject ? `Subject: ${subject}` : "No subject given -- suggest one, see the Subject: line rule."}
---
${steeringPrompt ? `Instructions from the user: ${steeringPrompt}` : "No specific instructions were given -- write a short, polite placeholder email and make clear in the text that it still needs details from the user."}${ragContext}${ownStyleContext}${historyContext}

Write the email now.`;

  return [
    { role: "system", content: NEW_EMAIL_SYSTEM_PROMPT },
    { role: "user", content: userPrompt },
  ];
}

/**
 * Splits a leading "Subject: ..." line off model output, but only when the caller left the
 * subject blank: a subject the user typed is authoritative, and the model was instructed not to
 * add its own in that case anyway. Shared by the non-streaming generateNewEmail and the
 * streaming runStreamed so both paths agree exactly.
 */
function splitSubject(fullText, userSubject) {
  const text = fullText.trim();
  if (!userSubject) {
    // Match on the TRIMMED text: a model that pads its output with a leading newline would
    // otherwise escape the anchored "Subject:" match.
    const match = text.match(/^Subject:\s*(.+?)\s*\n+([\s\S]*)$/i);
    if (match) {
      return { draft: match[2].trim(), subject: match[1].trim() };
    }
  }
  return { draft: text, subject: userSubject || "" };
}

/**
 * Composes a brand-new email (no originating message), from the user's instructions alone.
 * Returns { draft, subject }: if the caller left `subject` blank, a leading "Subject: ..." line
 * the model was asked to produce is split off and returned separately, never left in the body.
 */
async function generateNewEmail({ to, subject, steeringPrompt, backend, signal }) {
  const settings = await getSettings();
  const messages = await buildNewEmailMessages(to, subject, steeringPrompt);
  const raw = backend === "remote"
    ? await callLiteLLM(settings, messages, signal)
    : await callOllama(settings, messages, signal);

  return splitSubject(raw, subject);
}

/*
 * Streaming entry point, used by the UI over a long-lived runtime port (browser.runtime.connect
 * with name "mail-agent-stream") instead of the one-shot runtime.sendMessage the non-streaming
 * path uses -- a port stays open across as many round-trips as the draft needs.
 *
 * Protocol (one message in, many out):
 *   in : { type:"start", action:"generateDraft"|"generateNewEmail", ...payload, backend }
 *   in : { type:"stop" } at any time -- aborts the in-flight request
 *   out: { type:"token", text } -- repeated, one per non-empty model token
 *        then exactly one of:
 *        { type:"retrying" } -- a pre-first-token failure, falling back to the non-streaming call
 *        { type:"done", draft[, subject] } -- subject present for "generateNewEmail"
 *        { type:"error", error }
 * A voluntary stop (Stop button or the agent window closing) sends NOTHING after it: the user
 * asked to be left alone, so silence is the answer, and no fallback is attempted either.
 *
 * Every send goes through safePost because the port can be disconnected at the very moment an
 * abort lands (or right after we send) -- port.postMessage then throws, and that must never
 * mask the real outcome of the request.
 */
async function runStreamed(port, action, payload) {
  const ac = new AbortController();
  let tokensEmitted = 0;
  let disconnected = false;
  let stopped = false;

  const safePost = (msg) => {
    try {
      port.postMessage(msg);
    } catch (err) {
      // The UI is gone (window closed / port disconnected) -- there is nobody left to tell.
    }
  };

  // Closing the agent window (onDisconnect) and the UI sending { type:"stop" } both mean "stop
  // now": aborting makes reader.read() inside streamOllama/streamLiteLLM reject with AbortError,
  // which nextStepOnError maps to "abort" -- a silent return, no fallback, no scary error.
  // `disconnected` is remembered separately because, after the abort is already over, the
  // fallback path must also skip the non-streaming retry: nobody is watching it, and on a local
  // CPU model it would still burn several minutes of compute for no one.
  port.onDisconnect.addListener(() => {
    disconnected = true;
    ac.abort();
  });
  port.onMessage.addListener((msg) => {
    if (msg && msg.type === "stop") {
      // Remembered as well: the non-streaming fallback (below) is abortable through the same
      // signal, but a stop that lands in the gap between "decide to fall back" and "the fetch
      // actually starts" is still possible -- and the post-checks below drop the result either
      // way, so no draft the user just asked to stop can ever be shown.
      stopped = true;
      ac.abort();
    }
  });

  const onToken = (text) => {
    tokensEmitted += 1;
    safePost({ type: "token", text });
  };

  try {
    // Building the prompt (RAG search included) can itself fail -- that is still a
    // pre-first-token failure, i.e. the same "fallback" case as a request that never produced
    // a single token.
    const settings = await getSettings();
    const messages = action === "generateNewEmail"
      ? await buildNewEmailMessages(payload.to, payload.subject, payload.steeringPrompt)
      : await buildDraftMessages(payload.email, payload.steeringPrompt);
    const full = payload.backend === "remote"
      ? await streamLiteLLM(settings, messages, onToken, ac.signal)
      : await streamOllama(settings, messages, onToken, ac.signal);

    // Same post-processing as the non-streaming path (generateDraft trims; generateNewEmail
    // additionally splits off a leading "Subject: ..." line when the user left one blank) --
    // through the SAME splitSubject() both paths now share, so they cannot drift apart.
    if (action === "generateNewEmail") {
      const { draft, subject } = splitSubject(full, payload.subject);
      safePost({ type: "done", draft, subject });
    } else {
      safePost({ type: "done", draft: full.trim() });
    }
  } catch (err) {
    switch (nextStepOnError(err, tokensEmitted)) {
      case "abort":
        // Deliberate stop -- nothing to say and, above all, nothing to re-run.
        return;
      case "fallback": {
        // Nothing is on screen yet, so the non-streaming retry is invisible except a status
        // line -- unless the user already left (Stop button or the window closing): in that
        // case launching a full LLM call nobody will ever read would be a pure waste of local
        // CPU, so skip it before it even starts.
        if (stopped || disconnected) return;
        safePost({ type: "retrying" });
        // The fallback is now abortable through the same signal: a Stop that lands WHILE it
        // runs aborts the fetch itself (AbortError) instead of finishing an unwanted call.
        const fallbackCall = action === "generateNewEmail"
          ? () => generateNewEmail({ ...payload, signal: ac.signal })
          : () => generateDraft({ ...payload, signal: ac.signal });
        try {
          const result = await fallbackCall();
          if (stopped || disconnected) return; // the user stopped mid-fallback: drop the result
          safePost(action === "generateNewEmail" ? { type: "done", ...result } : { type: "done", draft: result });
        } catch (fallbackErr) {
          // An AbortError here means a stop/disconnect landed while the fallback ran: that is a
          // deliberate stop, not a failure -- silence remains the answer.
          if (stopped || disconnected || (fallbackErr && fallbackErr.name === "AbortError")) return;
          // Both paths failed: say so with BOTH reasons, so the user can act on the real one.
          safePost({
            type: "error",
            error: `Streaming unavailable (${err.message || err}) -- and the non-streaming fallback failed too: ${fallbackErr.message || fallbackErr}`,
          });
        }
        return;
      }
      default:
        // Tokens are already on screen -- a re-run would duplicate the partial text, so keep it
        // and simply report why the stream stopped.
        safePost({ type: "error", error: err.message || String(err) });
    }
  }
}

async function recordHistory({ subject, steeringPrompt, draft }) {
  const settings = await getSettings();
  const history = [...settings.history, { timestamp: Date.now(), subject, steeringPrompt, draft }];
  await saveSetting({ history: history.slice(-MAX_HISTORY) });
}

async function recordNewEmailHistory({ subject, steeringPrompt, draft }) {
  const settings = await getSettings();
  const newEmailHistory = [...settings.newEmailHistory, { timestamp: Date.now(), subject, steeringPrompt, draft }];
  await saveSetting({ newEmailHistory: newEmailHistory.slice(-MAX_HISTORY) });
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function draftToHtml(draftText) {
  return draftText
    .split(/\n\s*\n/)
    .map((para) => `<p>${escapeHtml(para).replace(/\n/g, "<br>")}</p>`)
    .join("");
}

/**
 * setComposeDetails({body}) REPLACES the whole body, so we first read what Thunderbird already
 * put there (quoted original + the user's signature for a reply; just the signature for a new
 * email) and prepend our draft to it. The editor can still be empty for a moment right after the
 * compose tab opens, hence the short retry loop.
 */
async function insertDraftIntoComposeTab(composeTabId, draftText) {
  let details = await browser.compose.getComposeDetails(composeTabId);
  for (let i = 0; i < 10 && !(details.isPlainText ? details.plainTextBody : details.body); i++) {
    await new Promise((r) => setTimeout(r, 200));
    details = await browser.compose.getComposeDetails(composeTabId);
  }

  if (details.isPlainText) {
    await browser.compose.setComposeDetails(composeTabId, {
      plainTextBody: `${draftText}\n\n${details.plainTextBody || ""}`,
    });
  } else {
    const existing = details.body || "<html><body></body></html>";
    const draftHtml = draftToHtml(draftText);
    const body = /<body[^>]*>/i.test(existing)
      ? existing.replace(/<body[^>]*>/i, (tag) => `${tag}${draftHtml}<br>`)
      : draftHtml + existing;
    await browser.compose.setComposeDetails(composeTabId, { body });
  }
}

async function insertIntoReply({ messageId, draftText }) {
  const composeTab = await browser.compose.beginReply(messageId, "replyToSender");
  await insertDraftIntoComposeTab(composeTab.id, draftText);
  return composeTab.id;
}

async function insertIntoNewCompose({ to, subject, draftText }) {
  const recipients = (to || "")
    .split(",")
    .map((r) => r.trim())
    .filter(Boolean);
  const composeTab = await browser.compose.beginNew({
    to: recipients,
    subject: subject || "",
  });
  await insertDraftIntoComposeTab(composeTab.id, draftText);
  return composeTab.id;
}

/*
 * Ollama rejects cross-origin requests whose Origin it doesn't know (HTTP 403), and extension
 * requests carry "Origin: moz-extension://<uuid>". Rewriting it to Ollama's own origin makes the
 * local backend work without the user having to set OLLAMA_ORIGINS. Only requests to localhost /
 * 127.0.0.1 (our host permissions) are touched.
 */
browser.webRequest.onBeforeSendHeaders.addListener(
  (details) => {
    if (!details.originUrl || !details.originUrl.startsWith("moz-extension://")) return {};
    const target = new URL(details.url).origin;
    for (const header of details.requestHeaders) {
      if (header.name.toLowerCase() === "origin") header.value = target;
    }
    return { requestHeaders: details.requestHeaders };
  },
  { urls: ["http://localhost/*", "http://127.0.0.1/*"] },
  ["blocking", "requestHeaders"]
);

async function handleMessage(message) {
  switch (message.action) {
    case "getDisplayedEmail":
      return getDisplayedEmail(message.tabId);
    case "getSettings":
      return getSettings();
    case "saveSettings":
      await saveSetting(message.settings);
      await Rag.onSettingsChanged(message.settings);
      return null;
    case "generateDraft":
      return { draft: await generateDraft(message) };
    case "acceptDraft": {
      const composeTabId = await insertIntoReply(message);
      await recordHistory(message);
      return { composeTabId };
    }
    case "generateNewEmail":
      return generateNewEmail(message);
    case "acceptNewEmail": {
      const composeTabId = await insertIntoNewCompose(message);
      await recordNewEmailHistory(message);
      return { composeTabId };
    }
    case "ragListAccounts":
      return Rag.listAccounts();
    case "ragListFolders":
      return Rag.listFolders(message.accountId);
    case "ragGetStatus":
      return Rag.getStatus();
    case "ragPullNow":
      return Rag.pullNow();
    case "ragClearIndex":
      return Rag.clearIndex();
    default:
      throw new Error(`Unknown action: ${message.action}`);
  }
}

// Returning a Promise from the listener is how Thunderbird/Firefox deliver an async response.
browser.runtime.onMessage.addListener((message) =>
  handleMessage(message).then(
    (data) => ({ ok: true, data }),
    (err) => ({ ok: false, error: err.message || String(err) })
  )
);

/*
 * Streaming path (see runStreamed for the protocol). The UI connects with
 * browser.runtime.connect("mail-agent-stream") and must send its { type:"start", ... } message
 * first; we wait for exactly that before starting the work, so prompt-building errors (RAG
 * search failing) are handled inside runStreamed's error logic instead of here. Anything else
 * first -- or a port that never sends anything -- is a protocol error, and closing the port is
 * the only sane reaction to it. This is the extension's only onConnect listener.
 *
 * At most ONE generation runs at a time, enforced HERE (background-side) and not just in the
 * popup: the popup's own isGenerating guard only covers its own window, but a second agent
 * window (an orphaned one from a crashed session, plus the toolbar one) is a separate page that
 * can legitimately connect while the first is still generating -- two concurrent LLM streams
 * would interleave badly and the second would race for the compose APIs. The guard is a module
 * variable set when a session is accepted and cleared in the .finally() wrapping runStreamed in
 * the listener below (which fires on EVERY outcome: done, error, abort, disconnect), so
 * "activeStreamPort !== null" is exactly "a session is in flight" -- no separate liveness
 * probe is possible or needed.
 */
let activeStreamPort = null;

browser.runtime.onConnect.addListener((port) => {
  if (port.name !== "mail-agent-stream") return;
  const firstMessage = new Promise((resolve, reject) => {
    const listener = (msg) => {
      port.onMessage.removeListener(listener);
      port.onDisconnect.removeListener(disconnected);
      resolve(msg);
    };
    const disconnected = () => {
      // The UI left before ever sending start -- nothing to start, and no one left to report it to.
      port.onMessage.removeListener(listener);
      reject(new Error("streaming port disconnected before its start message"));
    };
    port.onMessage.addListener(listener);
    port.onDisconnect.addListener(disconnected);
  });
  firstMessage
    .then((msg) => {
      if (!msg || msg.type !== "start") {
        try { port.disconnect(); } catch (err) { /* port may already be gone */ }
        return null;
      }
      if (activeStreamPort !== null) {
        // Another generation is already running (from another agent window): refuse this one
        // instead of starting a second concurrent LLM stream. The refusal goes out BEFORE the
        // disconnect so the UI gets a reason rather than a bare "disconnected".
        try {
          port.postMessage({ type: "error", error: "Another generation is already running -- stop it first." });
        } catch (err) { /* port may already be gone */ }
        try { port.disconnect(); } catch (err) { /* port may already be gone */ }
        return null;
      }
      activeStreamPort = port;
      return runStreamed(port, msg.action, msg).finally(() => {
        activeStreamPort = null; // released on every path: done, error, abort, and disconnect
      });
    })
    .catch((err) => console.error("LaboBots: streaming session did not complete", err));
});

Rag.init().catch((err) => console.error("LaboBots RAG: init failed", err));

// The toolbar button no longer opens a transient popup (see manifest.json) -- it opens the UI as
// its own standalone window instead, specifically so generating a draft no longer blocks the
// rest of Thunderbird: a popup closes (and loses its in-flight work) the moment it loses focus, a
// separate window doesn't. (Thunderbird has no sidebarAction API -- that's Firefox-only -- so a
// docked panel isn't an option here.) Clicking the button again just refocuses the existing
// window rather than opening a second one.
let agentWindowId = null;

browser.browserAction.onClicked.addListener(async () => {
  if (agentWindowId !== null) {
    try {
      await browser.windows.update(agentWindowId, { focused: true });
      return;
    } catch (err) {
      agentWindowId = null; // the window was closed without us noticing -- fall through and reopen
    }
  }
  const win = await browser.windows.create({
    url: browser.runtime.getURL("popup/popup.html"),
    type: "popup",
    width: 420,
    height: 680,
  });
  agentWindowId = win.id;
});

browser.windows.onRemoved.addListener((windowId) => {
  if (windowId === agentWindowId) agentWindowId = null;
});
