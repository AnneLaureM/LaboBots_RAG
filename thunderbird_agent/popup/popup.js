/*
 * LaboBots Mail Agent -- agent window script. Pure UI glue: everything that actually talks to
 * the LLM or to Thunderbird's compose API lives in background.js (see the comment at its top for
 * why). Opened by background.js as its own standalone browser.windows.create() window (Thunderbird
 * has no sidebarAction API, unlike Firefox) rather than a toolbar popup, specifically so a draft
 * can take its time generating without blocking the rest of Thunderbird: a popup closes -- losing
 * its in-flight work -- the moment it loses focus, this separate window doesn't, and the main
 * Thunderbird window stays fully usable (reading/replying to other mail) while it works.
 *
 * Because this is its own window, "the currently displayed email" is never this window's own
 * active tab -- it has none -- but whatever tab is active in Thunderbird's own main window(s).
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
let isGenerating = false; // guards against the "which email is this for" tracker below jumping to
                           // a different email mid-generation -- see refreshCurrentEmail()

function send(action, extra = {}) {
  return browser.runtime.sendMessage({ action, ...extra });
}

// Finds the active tab in Thunderbird's own main ("normal") window -- never this agent window
// itself, which is a type:"popup" window with no mail tabs of its own. Prefers whichever main
// window was focused most recently, in case the user has more than one open.
async function getMainWindowActiveTab() {
  const windows = await browser.windows.getAll({ windowTypes: ["normal"] });
  const sorted = [...windows].sort((a, b) => (b.focused ? 1 : 0) - (a.focused ? 1 : 0));
  for (const win of sorted) {
    const [tab] = await browser.tabs.query({ windowId: win.id, active: true });
    if (tab) return tab;
  }
  return null;
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

/*
 * Unlike the old toolbar popup (torn down and rebuilt every time it opened), this window stays
 * open as the user clicks around Thunderbird's main window -- so instead of reading "the tab this
 * was opened from" once at startup, it has to track which message is currently displayed there as
 * that changes. Skipped while a generation is in flight so the reply/insert flow can't end up
 * pointed at a different email than the one the draft was actually written for.
 */
async function refreshCurrentEmail() {
  if (isGenerating) return;

  const tab = await getMainWindowActiveTab();
  const emailResp = await send("getDisplayedEmail", { tabId: tab && tab.id });
  draftAreaEl.hidden = true;
  setStatus("");

  if (!emailResp.ok) {
    // No message displayed (e.g. no email selected, or a non-mail tab is focused) -- there is
    // nothing to reply to right now.
    currentEmail = null;
    emailSummaryEl.textContent = emailResp.error;
    generateBtn.disabled = true;
    return;
  }
  currentEmail = emailResp.data;
  generateBtn.disabled = false;
  emailSummaryEl.innerHTML =
    `<strong>${escapeHtml(currentEmail.subject)}</strong><br>from ${escapeHtml(currentEmail.from)}` +
    attachmentsSummaryHtml(currentEmail.attachments);
}

async function init() {
  const settingsResp = await send("getSettings");
  if (settingsResp.ok) {
    backendEl.value = settingsResp.data.backend;
    newBackendEl.value = settingsResp.data.backend;
  }

  await refreshCurrentEmail();

  // Keep following the user as they click around other emails/folders in Thunderbird's main
  // window while this agent window stays open.
  browser.messageDisplay.onMessageDisplayed.addListener(() => refreshCurrentEmail());
  browser.tabs.onActivated.addListener(() => refreshCurrentEmail());
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
  isGenerating = true;
  generateBtn.disabled = true;
  regenerateBtn.disabled = true;
  draftAreaEl.hidden = true;
  setStatus("Generating draft... this can take a while on a local CPU model -- feel free to read or reply to other emails in the meantime, this panel will keep working.");

  const backend = backendEl.value;
  browser.storage.local.set({ backend });

  const resp = await send("generateDraft", {
    email: currentEmail,
    steeringPrompt: steeringEl.value.trim(),
    backend,
  });

  isGenerating = false;
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

  // The reply is now a real compose tab -- nothing more to do here. Unlike the old popup, this
  // panel doesn't close itself: just clear the draft and report success, ready for the next email.
  draftAreaEl.hidden = true;
  steeringEl.value = "";
  setStatus("Inserted into a new reply window.");
}

async function generateNew() {
  isGenerating = true;
  newGenerateBtn.disabled = true;
  newRegenerateBtn.disabled = true;
  newDraftAreaEl.hidden = true;
  setNewStatus("Generating draft... this can take a while on a local CPU model -- feel free to read or reply to other emails in the meantime, this panel will keep working.");

  const backend = newBackendEl.value;
  browser.storage.local.set({ backend });

  const resp = await send("generateNewEmail", {
    to: newToEl.value.trim(),
    subject: newSubjectEl.value.trim(),
    steeringPrompt: newSteeringEl.value.trim(),
    backend,
  });

  isGenerating = false;
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

  // Same as insertReply(): the email is now a real compose tab, and this panel stays open.
  newDraftAreaEl.hidden = true;
  newToEl.value = "";
  newSubjectEl.value = "";
  newSteeringEl.value = "";
  setNewStatus("Opened in a new compose window.");
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
