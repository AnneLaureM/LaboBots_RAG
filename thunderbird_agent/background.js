/*
 * LaboBots Mail Agent -- background script.
 *
 * Owns everything the popup itself shouldn't: reading the displayed message, calling the LLM
 * (local Ollama or remote LiteLLM proxy -- same two backends as notebooks 1 and 2), keeping a
 * small local "writing style" history, and inserting the generated draft into a REAL Thunderbird
 * reply window (via compose.beginReply) so the user's signature and Send button are the native
 * ones -- this extension never sends anything itself.
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
};

const MAX_HISTORY = 20;
const MAX_HISTORY_IN_PROMPT = 3; // how many past drafts get folded into the prompt as style context

const DRAFT_SYSTEM_PROMPT = `You are an email-drafting assistant. You will be shown an email and
asked to draft a reply to it. Rules:
- Reply in the same language as the original email.
- Write ONLY the body of the reply -- no "Dear X," greeting line and no sign-off/signature. The
  user's own Thunderbird signature is added automatically after your text; do not duplicate it.
- Follow the user's own steering instructions if given (tone, what to say, what to avoid, length).
- Stay concise and professional unless the instructions say otherwise.
- If reference is made to "previous replies" below, they are only style examples (how this
  person usually writes) -- never repeat their content, just match the tone/register.`;

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

async function getDisplayedEmail(tabId) {
  // The popup passes the id of the tab it was opened from: the background page has no "current
  // tab" of its own, and getDisplayedMessage() needs to know which tab's message to read.
  const message = await browser.messageDisplay.getDisplayedMessage(tabId);
  if (!message) {
    throw new Error("No single message is currently displayed. Open (or select) one email first.");
  }
  const body = await getMessageBody(message.id);
  return {
    messageId: message.id,
    subject: message.subject,
    from: message.author,
    body: body.slice(0, 6000), // keep the prompt a reasonable size for a small local model
  };
}

function buildHistoryContext(history) {
  const recent = history.slice(-MAX_HISTORY_IN_PROMPT);
  if (recent.length === 0) return "";
  const examples = recent
    .map((h, i) => `Previous reply example ${i + 1} (style reference only):\n${h.draft}`)
    .join("\n\n");
  return `\n\nHere are a few of this user's own previous replies, for style reference only:\n\n${examples}`;
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

async function callOllama(settings, messages) {
  const resp = await fetchOrExplain(settings.ollama_url, "Cannot reach Ollama -- is 'ollama serve' running?", `${settings.ollama_url}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: settings.ollama_model, messages, stream: false }),
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

async function callLiteLLM(settings, messages) {
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
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`LiteLLM proxy returned HTTP ${resp.status}. ${text} -- is the SSH tunnel to the remote host open (manage_remote_rag.sh tunnel)?`);
  }
  const data = await resp.json();
  return data.choices[0].message.content;
}

async function generateDraft({ email, steeringPrompt, backend }) {
  const settings = await getSettings();
  const historyContext = buildHistoryContext(settings.history);

  const userPrompt = `Original email
From: ${email.from}
Subject: ${email.subject}

${email.body}

---
${steeringPrompt ? `Steering instructions from the user: ${steeringPrompt}` : "No specific steering instructions -- use your best judgment for a reasonable reply."}${historyContext}

Draft the reply now.`;

  const messages = [
    { role: "system", content: DRAFT_SYSTEM_PROMPT },
    { role: "user", content: userPrompt },
  ];

  const draft = backend === "remote"
    ? await callLiteLLM(settings, messages)
    : await callOllama(settings, messages);

  return draft.trim();
}

async function recordHistory({ subject, steeringPrompt, draft }) {
  const settings = await getSettings();
  const history = [...settings.history, { timestamp: Date.now(), subject, steeringPrompt, draft }];
  await saveSetting({ history: history.slice(-MAX_HISTORY) });
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

async function insertIntoReply({ messageId, draftText }) {
  const composeTab = await browser.compose.beginReply(messageId, "replyToSender");
  // setComposeDetails({body}) REPLACES the whole body, so we first read what Thunderbird put
  // there (quoted original + the user's signature) and prepend our draft to it. The editor can
  // still be empty for a moment right after beginReply resolves, hence the short retry loop.
  let details = await browser.compose.getComposeDetails(composeTab.id);
  for (let i = 0; i < 10 && !(details.isPlainText ? details.plainTextBody : details.body); i++) {
    await new Promise((r) => setTimeout(r, 200));
    details = await browser.compose.getComposeDetails(composeTab.id);
  }

  if (details.isPlainText) {
    await browser.compose.setComposeDetails(composeTab.id, {
      plainTextBody: `${draftText}\n\n${details.plainTextBody || ""}`,
    });
  } else {
    const existing = details.body || "<html><body></body></html>";
    const draftHtml = draftToHtml(draftText);
    const body = /<body[^>]*>/i.test(existing)
      ? existing.replace(/<body[^>]*>/i, (tag) => `${tag}${draftHtml}<br>`)
      : draftHtml + existing;
    await browser.compose.setComposeDetails(composeTab.id, { body });
  }
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
      return null;
    case "generateDraft":
      return { draft: await generateDraft(message) };
    case "acceptDraft": {
      const composeTabId = await insertIntoReply(message);
      await recordHistory(message);
      return { composeTabId };
    }
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
