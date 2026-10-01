/*
 * LaboBots Mail Agent -- popup script. Pure UI glue: everything that actually talks to the LLM
 * or to Thunderbird's compose API lives in background.js (see the comment at its top for why).
 */

const emailSummaryEl = document.getElementById("email-summary");
const backendEl = document.getElementById("backend");
const steeringEl = document.getElementById("steering");
const generateBtn = document.getElementById("generate-btn");
const statusEl = document.getElementById("status");
const draftAreaEl = document.getElementById("draft-area");
const draftTextEl = document.getElementById("draft-text");
const regenerateBtn = document.getElementById("regenerate-btn");
const insertBtn = document.getElementById("insert-btn");
const optionsLink = document.getElementById("options-link");

const tabReplyEl = document.getElementById("tab-reply");
const tabNewEl = document.getElementById("tab-new");
const replySectionEl = document.getElementById("reply-section");
const newSectionEl = document.getElementById("new-section");

const newToEl = document.getElementById("new-to");
const newSubjectEl = document.getElementById("new-subject");
const newBackendEl = document.getElementById("new-backend");
const newSteeringEl = document.getElementById("new-steering");
const newGenerateBtn = document.getElementById("new-generate-btn");
const newStatusEl = document.getElementById("new-status");
const newDraftAreaEl = document.getElementById("new-draft-area");
const newDraftSubjectEl = document.getElementById("new-draft-subject");
const newDraftTextEl = document.getElementById("new-draft-text");
const newRegenerateBtn = document.getElementById("new-regenerate-btn");
const newInsertBtn = document.getElementById("new-insert-btn");

let currentEmail = null;

function send(action, extra = {}) {
  return browser.runtime.sendMessage({ action, ...extra });
}

function setStatus(text, isError = false) {
  statusEl.hidden = !text;
  statusEl.textContent = text;
  statusEl.style.color = isError ? "#B91C1C" : "";
}

function setNewStatus(text, isError = false) {
  newStatusEl.hidden = !text;
  newStatusEl.textContent = text;
  newStatusEl.style.color = isError ? "#B91C1C" : "";
}

function showTab(tab) {
  const isReply = tab === "reply";
  tabReplyEl.classList.toggle("active", isReply);
  tabNewEl.classList.toggle("active", !isReply);
  replySectionEl.hidden = !isReply;
  newSectionEl.hidden = isReply;
}

async function init() {
  const settingsResp = await send("getSettings");
  if (settingsResp.ok) {
    backendEl.value = settingsResp.data.backend;
    newBackendEl.value = settingsResp.data.backend;
  }

  // The popup knows which tab it was opened from; the background page doesn't.
  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  const emailResp = await send("getDisplayedEmail", { tabId: tab && tab.id });
  if (!emailResp.ok) {
    // No message displayed (e.g. opened from the permanent toolbar button rather than a message
    // view) -- there is nothing to reply to, so go straight to "New email" and hide the Reply tab.
    emailSummaryEl.textContent = emailResp.error;
    generateBtn.disabled = true;
    tabReplyEl.disabled = true;
    showTab("new");
    return;
  }
  currentEmail = emailResp.data;
  emailSummaryEl.innerHTML =
    `<strong>${escapeHtml(currentEmail.subject)}</strong><br>from ${escapeHtml(currentEmail.from)}` +
    attachmentsSummaryHtml(currentEmail.attachments);
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function attachmentsSummaryHtml(attachments) {
  if (!attachments || attachments.length === 0) return "";
  const items = attachments
    .map((a) => {
      const icon = a.textIncluded ? "📄" : a.contentType.startsWith("image/") ? "🖼️" : "📎";
      const status = a.textIncluded ? "content included" : a.note || "not read";
      return `<li>${icon} ${escapeHtml(a.name)} <span class="muted">(${escapeHtml(status)})</span></li>`;
    })
    .join("");
  return `<ul class="attachments">${items}</ul>`;
}

async function generate() {
  generateBtn.disabled = true;
  regenerateBtn.disabled = true;
  draftAreaEl.hidden = true;
  setStatus("Generating draft... this can take a while on a local CPU model.");

  const backend = backendEl.value;
  browser.storage.local.set({ backend });

  const resp = await send("generateDraft", {
    email: currentEmail,
    steeringPrompt: steeringEl.value.trim(),
    backend,
  });

  generateBtn.disabled = false;
  regenerateBtn.disabled = false;

  if (!resp.ok) {
    setStatus(resp.error, true);
    return;
  }

  setStatus("");
  draftTextEl.value = resp.data.draft;
  draftAreaEl.hidden = false;
}

async function insertReply() {
  insertBtn.disabled = true;
  setStatus("Opening the reply window...");

  const resp = await send("acceptDraft", {
    messageId: currentEmail.messageId,
    draftText: draftTextEl.value,
    subject: currentEmail.subject,
    steeringPrompt: steeringEl.value.trim(),
    draft: draftTextEl.value,
  });

  insertBtn.disabled = false;

  if (!resp.ok) {
    setStatus(resp.error, true);
    return;
  }

  window.close(); // the reply is now a real compose tab; nothing more for this popup to do
}

async function generateNew() {
  newGenerateBtn.disabled = true;
  newRegenerateBtn.disabled = true;
  newDraftAreaEl.hidden = true;
  setNewStatus("Generating draft... this can take a while on a local CPU model.");

  const backend = newBackendEl.value;
  browser.storage.local.set({ backend });

  const resp = await send("generateNewEmail", {
    to: newToEl.value.trim(),
    subject: newSubjectEl.value.trim(),
    steeringPrompt: newSteeringEl.value.trim(),
    backend,
  });

  newGenerateBtn.disabled = false;
  newRegenerateBtn.disabled = false;

  if (!resp.ok) {
    setNewStatus(resp.error, true);
    return;
  }

  setNewStatus("");
  newDraftSubjectEl.value = resp.data.subject;
  newDraftTextEl.value = resp.data.draft;
  newDraftAreaEl.hidden = false;
}

async function insertNewEmail() {
  newInsertBtn.disabled = true;
  setNewStatus("Opening the compose window...");

  const resp = await send("acceptNewEmail", {
    to: newToEl.value.trim(),
    subject: newDraftSubjectEl.value.trim(),
    draftText: newDraftTextEl.value,
    steeringPrompt: newSteeringEl.value.trim(),
    draft: newDraftTextEl.value,
  });

  newInsertBtn.disabled = false;

  if (!resp.ok) {
    setNewStatus(resp.error, true);
    return;
  }

  window.close(); // the email is now a real compose tab; nothing more for this popup to do
}

generateBtn.addEventListener("click", generate);
regenerateBtn.addEventListener("click", generate);
insertBtn.addEventListener("click", insertReply);
newGenerateBtn.addEventListener("click", generateNew);
newRegenerateBtn.addEventListener("click", generateNew);
newInsertBtn.addEventListener("click", insertNewEmail);
tabReplyEl.addEventListener("click", () => showTab("reply"));
tabNewEl.addEventListener("click", () => showTab("new"));
optionsLink.addEventListener("click", (e) => {
  e.preventDefault();
  browser.runtime.openOptionsPage();
});

init();
