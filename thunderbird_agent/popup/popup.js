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
const stopBtn = document.getElementById("stop-btn");
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
const newStopBtn = document.getElementById("new-stop-btn");

/*
 * The elements a streaming session needs to drive, grouped per tab so beginStream() and
 * finalizeSession() can lock down and restore BOTH tabs from one place instead of repeating
 * the button logic in every state transition.
 */
const replyUI = {
  status: setStatus,
  generate: generateBtn,
  regenerate: regenerateBtn,
  insert: insertBtn,
  stop: stopBtn,
  area: draftAreaEl,
  textarea: draftTextEl,
  subjectInput: null, // the reply tab has no subject field of its own
};
const newUI = {
  status: setNewStatus,
  generate: newGenerateBtn,
  regenerate: newRegenerateBtn,
  insert: newInsertBtn,
  stop: newStopBtn,
  area: newDraftAreaEl,
  textarea: newDraftTextEl,
  subjectInput: newDraftSubjectEl,
};

let currentEmail = null;
let isGenerating = false; // guards against the "which email is this for" tracker below jumping to
                           // a different email mid-generation -- see refreshCurrentEmail()
let activeSession = null; // the in-flight streaming generation (below), null when idle

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

  try {
    const tab = await getMainWindowActiveTab();
    const emailResp = await send("getDisplayedEmail", { tabId: tab && tab.id });
    // Hide the draft area only when it is EMPTY: this function also runs on onActivated /
    // onMessageDisplayed while the user simply clicks around Thunderbird's main window, and an
    // unconditional hide would swallow a draft (partial or complete) the user stopped or kept
    // from an earlier generation. A fresh generation is unaffected: beginStream() clears the
    // textarea and forces the area visible before the first token.
    if (draftTextEl.value.trim() === "") draftAreaEl.hidden = true;
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
  } catch (err) {
    currentEmail = null;
    emailSummaryEl.textContent = `Unexpected error: ${err.message || err}`;
    generateBtn.disabled = true;
  }
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

/*
 * Streaming generation -- replaces the old one-shot send("generateDraft") / send("generateNewEmail"):
 * the background now streams tokens over a long-lived port instead of answering a single
 * runtime.sendMessage, so the draft fills in as the model produces it (protocol: background.js,
 * runStreamed() and the onConnect listener at the end of that file).
 *
 * A generation is one session shared by BOTH tabs: beginStream() locks the controls of the other
 * tab as well (its Generate/Regenerate/Insert can't race the in-flight one, and its Stop button
 * aborts the very same stream), and finalizeSession() is the single place that restores both
 * tabs. Every terminal outcome -- done, error, stop, the background dying mid-flight, even the
 * port never being established -- funnels through it, so no path is left with the buttons locked
 * and the guard still set.
 */

/*
 * Live progress for the generating tab's status line: the running token count, the average
 * tok/s and the elapsed time, refreshed on every token AND on a ~500 ms timer. The timer is the
 * point during the initial silence (a CPU model can take 10-60 s before the first token) and
 * again while the non-streaming fallback runs -- during both, the line must keep moving.
 *
 * One setInterval per session, owned by the session and cleared in finalizeSession() on EVERY
 * outcome (done / error / stop / disconnect / connect-failure). A leaked interval would repaint a
 * finalized line forever and burn CPU, so the tick's session.finalized guard plus the explicit
 * stopProgress() make any stray tick a no-op.
 *
 * `now` and the interval functions are module-level bindings (not inlined) so the Node test
 * harness can substitute a controllable clock and a recording interval; in the browser the
 * defaults are the real Date.now / setInterval / clearInterval, so behaviour is unchanged.
 */
let now = Date.now;
let setIntervalFn = setInterval;
let clearIntervalFn = clearInterval;

function elapsedMs(session) {
  return now() - session.startedAt;
}

function formatElapsed(ms) {
  return (ms / 1000).toFixed(1) + " s";
}

// Average rate over the whole session. Under 100 ms there is not enough time for a meaningful
// number (and the division would spike toward Infinity), so report "—" instead of a misleading rate.
function formatSpeed(tokens, ms) {
  if (ms < 100) return "— tok/s";
  return (tokens / (ms / 1000)).toFixed(1) + " tok/s";
}

function buildLiveStatus(session) {
  const ms = elapsedMs(session);
  const elapsed = formatElapsed(ms);
  if (session.fallbackMode) {
    // Keep the clock ticking while the fallback runs: without this the user goes blind again for
    // the entire (possibly long) non-streaming call.
    return `Streaming unavailable -- retrying as a single response (${elapsed})`;
  }
  if (session.tokenCount === 0) {
    return `Generating -- waiting for the first token (${elapsed})`;
  }
  return `Generating -- ${session.tokenCount} token${session.tokenCount > 1 ? "s" : ""} · ${formatSpeed(session.tokenCount, ms)} · ${elapsed}`;
}

function startProgress(session) {
  stopProgress(session); // idempotent: a session must never end up with two intervals
  session.progressTimer = setIntervalFn(() => {
    if (session.finalized) return; // a tick can outlive finalize() by a cycle -- make it inert
    session.ui.status(buildLiveStatus(session));
  }, 500);
}

function stopProgress(session) {
  if (session.progressTimer !== null) {
    clearIntervalFn(session.progressTimer);
    session.progressTimer = null;
  }
}

function beginStream(tab) {
  if (isGenerating) return null; // one generation at a time, shared by both tabs
  isGenerating = true;

  const ui = tab === "reply" ? replyUI : newUI;
  const otherUI = tab === "reply" ? newUI : replyUI;

  const session = {
    tab,
    ui,
    port: null,
    finalized: false, // stop/done/error/disconnect can all race -- the first one to win acts
    // Remembered so the other tab's Insert comes back exactly as it was: a previously complete
    // draft there must stay insertable, an incomplete one must not suddenly become one.
    otherInsertEnabled: !otherUI.insert.disabled,
    // Live progress (buildLiveStatus): the count is bumped per received token MESSAGE, never
    // derived from the text length (a token message may carry an empty or a multi-character piece).
    tokenCount: 0,
    startedAt: now(), // elapsed time is measured from generation start, not from the first token
    fallbackMode: false, // set true on "retrying" -- the base status line then switches over
    progressTimer: null, // the ~500 ms interval handle, cleared in finalizeSession()
  };
  activeSession = session;

  for (const t of [replyUI, newUI]) {
    t.generate.disabled = true;
    t.regenerate.disabled = true;
    t.insert.disabled = true; // never insert a draft that is still arriving
    t.stop.hidden = false;
  }

  // The generating tab starts with a cleared, immediately visible draft area: the first token has
  // a home as soon as it arrives, and an old draft from a previous generation can't be mistaken
  // for the new one.
  ui.textarea.value = "";
  if (ui.subjectInput) ui.subjectInput.value = "";
  ui.area.hidden = false;

  return session;
}

function handleStreamMessage(session, msg) {
  if (session.finalized) return; // late straggler (a token already in flight when we stopped)
  const ui = session.ui;

  switch (msg && msg.type) {
    case "token":
      // The model's text is untrusted: it is appended raw to a textarea's value and never
      // interpreted as HTML.
      ui.textarea.value += msg.text;
      ui.textarea.scrollTop = ui.textarea.scrollHeight; // keep the newest text in view
      session.tokenCount += 1; // count the token MESSAGE, not the characters it carries
      // Refresh right away -- don't make the user wait up to 500 ms for their own token to count.
      ui.status(buildLiveStatus(session));
      break;

    case "retrying":
      // Not an error: the background failed before the first token and is falling back to a
      // single non-streamed response -- the (empty) textarea is left exactly as it is.
      // Flip the base line (buildLiveStatus reads this) but KEEP the ~500 ms interval running:
      // the fallback can take a long time and the elapsed clock must keep moving so the user is
      // not blind again for its whole duration.
      session.fallbackMode = true;
      ui.status(buildLiveStatus(session));
      break;

    case "done":
      // The background's final, trimmed text is authoritative and can differ from the
      // concatenated tokens (for a new email, the "Subject:" line is split off over there).
      ui.textarea.value = msg.draft;
      if (ui.subjectInput) ui.subjectInput.value = msg.subject || "";
      finalizeSession(session, "done");
      break;

    case "error":
      // The partial text, if any, stays in the textarea: the user can edit it or Regenerate.
      finalizeSession(session, "error", msg.error || "The generation failed");
      break;

    default:
      // Unknown message type -- ignore it rather than corrupt the state machine.
      break;
  }
}

function handleStreamDisconnect(session) {
  // Fires both when the background genuinely dies mid-generation and after every finalize (we
  // disconnect the port ourselves there). finalizeSession() is idempotent, so only the former
  // case can still act.
  finalizeSession(session, "disconnected");
}

function finalizeSession(session, kind, detail) {
  if (session.finalized) return; // idempotent by design: stop/done/disconnect all race here
  session.finalized = true;
  activeSession = null;
  isGenerating = false; // released on EVERY outcome -- leaving it set would lock both tabs forever
  // Every outcome stops the per-session progress interval: done / error / stop / disconnect, and
  // even the connect-failure path that lands here with no interval to begin with (harmless no-op).
  stopProgress(session);

  if (session.port) {
    // Close our end of the port; whatever just happened, it is done talking to us. This may
    // re-dispatch this session's onDisconnect -- the guard above turns that into a no-op.
    try {
      session.port.disconnect();
    } catch (err) {
      // The port may already be gone (background crashed) -- there is nothing left to close.
    }
    session.port = null;
  }

  const genUI = session.ui;
  const otherUI = session.tab === "reply" ? newUI : replyUI;

  for (const t of [replyUI, newUI]) {
    t.generate.disabled = false;
    t.regenerate.disabled = false;
    t.stop.hidden = true;
  }

  // Insert is only meaningful for a COMPLETE draft: on the generating tab it comes back
  // exclusively after "done"; on the other tab it returns to whatever state it had before this
  // generation started.
  genUI.insert.disabled = kind !== "done";
  otherUI.insert.disabled = !session.otherInsertEnabled;

  // Terminal status: report the token count the user actually saw, so a stop or an error after
  // some streaming is not mistaken for "nothing happened".
  const kept = session.tokenCount > 0 ? ` -- ${session.tokenCount} token${session.tokenCount > 1 ? "s" : ""} kept` : "";
  if (kind === "done") {
    const ms = elapsedMs(session);
    if (session.fallbackMode) {
      // The fallback delivers one whole response, so reporting it as "0 tokens streamed" would
      // read like a failure -- say what actually happened instead.
      genUI.status(`Done -- delivered as a single response in ${formatElapsed(ms)}`);
    } else {
      genUI.status(`Done -- ${session.tokenCount} token${session.tokenCount > 1 ? "s" : ""} in ${formatElapsed(ms)} (${formatSpeed(session.tokenCount, ms)})`);
    }
  } else if (kind === "error") {
    genUI.status(detail + kept, true);
  } else if (kind === "stop") {
    genUI.status(session.tokenCount > 0 ? `Stopped -- ${session.tokenCount} token${session.tokenCount > 1 ? "s" : ""} kept` : "Stopped -- no tokens received yet");
  } else {
    genUI.status("The background process disconnected -- generation aborted" + kept, true);
  }
}

function startStreaming(tab, action, payload) {
  const session = beginStream(tab);
  if (!session) return; // the other tab is already generating -- both tabs' buttons are locked

  const backend = tab === "reply" ? backendEl.value : newBackendEl.value;
  // Remember the choice in storage for the popup's OWN init() on the next window open (the
  // background does NOT read it: the start message posted right below already carries the
  // backend). Fire-and-forget with the rejection swallowed on purpose -- a storage failure here
  // must neither block the generation that follows nor surface as an unhandled rejection.
  browser.storage.local.set({ backend }).catch(() => {});

  try {
    const port = browser.runtime.connect({ name: "mail-agent-stream" });
    session.port = port;
    port.onMessage.addListener((msg) => handleStreamMessage(session, msg));
    port.onDisconnect.addListener(() => handleStreamDisconnect(session));

    // Start the live progress: show the "waiting for first token" line immediately and tick it
    // every ~500 ms (a CPU model can take 10-60 s to start). finalizeSession() clears the timer.
    session.ui.status(buildLiveStatus(session));
    startProgress(session);
    // The "start" message must be the FIRST one on the port: the background waits for exactly
    // that before doing any work, and treats anything else as a protocol error. Its fields
    // (email / to / subject / steeringPrompt) are flat on the message, not nested.
    port.postMessage({ type: "start", action, ...payload, backend });
  } catch (err) {
    // connect() or the very first postMessage threw (e.g. the background page is restarting):
    // nothing will ever stream, so restore the UI instead of leaving every button locked.
    finalizeSession(session, "error", `Unexpected error: ${err.message || err}`);
  }
}

function generate() {
  startStreaming("reply", "generateDraft", {
    email: currentEmail,
    steeringPrompt: steeringEl.value.trim(),
  });
}

function stopGeneration() {
  const session = activeSession;
  if (!session || session.finalized) return; // idempotent: a double Stop click is a no-op
  // Tell the background first so it aborts; then settle locally. It sends nothing after a
  // voluntary stop, so there is nothing more to wait for.
  try {
    session.port.postMessage({ type: "stop" });
  } catch (err) {
    // The port is already gone -- the background is stopping (or stopped) on its own anyway.
  }
  finalizeSession(session, "stop");
}

async function insertReply() {
  insertBtn.disabled = true;
  setStatus("Opening the reply window...");

  try {
    const resp = await send("acceptDraft", {
      messageId: currentEmail.messageId,
      draftText: draftTextEl.value,
      subject: currentEmail.subject,
      steeringPrompt: steeringEl.value.trim(),
      draft: draftTextEl.value,
    });

    if (!resp.ok) {
      setStatus(resp.error, true);
      return;
    }

    // The reply is now a real compose tab -- nothing more to do here. Unlike the old popup, this
    // window doesn't close itself: just clear the draft and report success, ready for the next email.
    draftAreaEl.hidden = true;
    steeringEl.value = "";
    setStatus("Inserted into a new reply window.");
  } catch (err) {
    setStatus(`Unexpected error: ${err.message || err}`, true);
  } finally {
    insertBtn.disabled = false;
  }
}

function generateNew() {
  startStreaming("new", "generateNewEmail", {
    to: newToEl.value.trim(),
    subject: newSubjectEl.value.trim(),
    steeringPrompt: newSteeringEl.value.trim(),
  });
}

async function insertNewEmail() {
  newInsertBtn.disabled = true;
  setNewStatus("Opening the compose window...");

  try {
    const resp = await send("acceptNewEmail", {
      to: newToEl.value.trim(),
      subject: newDraftSubjectEl.value.trim(),
      draftText: newDraftTextEl.value,
      steeringPrompt: newSteeringEl.value.trim(),
      draft: newDraftTextEl.value,
    });

    if (!resp.ok) {
      setNewStatus(resp.error, true);
      return;
    }

    // Same as insertReply(): the email is now a real compose tab, and this window stays open.
    newDraftAreaEl.hidden = true;
    newToEl.value = "";
    newSubjectEl.value = "";
    newSteeringEl.value = "";
    setNewStatus("Opened in a new compose window.");
  } catch (err) {
    setNewStatus(`Unexpected error: ${err.message || err}`, true);
  } finally {
    newInsertBtn.disabled = false;
  }
}

generateBtn.addEventListener("click", generate);
regenerateBtn.addEventListener("click", generate);
insertBtn.addEventListener("click", insertReply);
stopBtn.addEventListener("click", stopGeneration);
newGenerateBtn.addEventListener("click", generateNew);
newRegenerateBtn.addEventListener("click", generateNew);
newInsertBtn.addEventListener("click", insertNewEmail);
newStopBtn.addEventListener("click", stopGeneration);
tabReplyEl.addEventListener("click", () => showTab("reply"));
tabNewEl.addEventListener("click", () => showTab("new"));
optionsLink.addEventListener("click", (e) => {
  e.preventDefault();
  browser.runtime.openOptionsPage();
});

init();

/*
 * Classic-script export so tests/stream.test.mjs can load this file under Node and drive the
 * state machine with fake messages and a controllable clock/interval. Inside the extension
 * `module` is undefined, so this block is inert and the names above stay plain globals.
 */
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    beginStream,
    handleStreamMessage,
    handleStreamDisconnect,
    finalizeSession,
    startStreaming,
    stopGeneration,
    // Test injection points (unused in production): swap the clock and the interval implementation.
    _setClock(fn) { now = fn; },
    _setIntervalImpl(start, stop) { setIntervalFn = start; clearIntervalFn = stop; },
  };
}
