const fields = ["ollama_url", "ollama_model", "litellm_url", "litellm_model", "litellm_key"];

function send(action, extra = {}) {
  return browser.runtime.sendMessage({ action, ...extra });
}

async function load() {
  const resp = await send("getSettings");
  if (!resp.ok) return;
  for (const key of fields) {
    document.getElementById(key).value = resp.data[key] ?? "";
  }
  const totalHistory = (resp.data.history || []).length + (resp.data.newEmailHistory || []).length;
  document.getElementById("history-count").textContent = totalHistory;

  document.getElementById("rag-enabled").checked = !!resp.data.ragEnabled;
  document.getElementById("rag-embed-model").value = resp.data.ragEmbedModel || "nomic-embed-text";
  document.getElementById("rag-pull-interval").value = resp.data.ragPullIntervalMinutes || 60;
  await loadRagAccounts(resp.data.ragAccountId || "");
  await refreshRagStatus();
}

async function loadRagAccounts(selectedId) {
  const select = document.getElementById("rag-account");
  const accountsResp = await send("ragListAccounts");
  if (accountsResp.ok) {
    for (const account of accountsResp.data) {
      const option = document.createElement("option");
      option.value = account.id;
      option.textContent = account.name;
      select.appendChild(option);
    }
  }
  select.value = selectedId;
}

async function save() {
  const settings = {};
  for (const key of fields) {
    settings[key] = document.getElementById(key).value.trim();
  }
  settings.ragEnabled = document.getElementById("rag-enabled").checked;
  settings.ragAccountId = document.getElementById("rag-account").value;
  settings.ragEmbedModel = document.getElementById("rag-embed-model").value.trim() || "nomic-embed-text";
  settings.ragPullIntervalMinutes = parseInt(document.getElementById("rag-pull-interval").value, 10) || 60;
  await send("saveSettings", { settings });

  const statusEl = document.getElementById("save-status");
  statusEl.hidden = false;
  statusEl.textContent = "Saved.";
  setTimeout(() => { statusEl.hidden = true; }, 2000);

  await refreshRagStatus();
}

async function clearHistory() {
  await send("saveSettings", { settings: { history: [], newEmailHistory: [] } });
  document.getElementById("history-count").textContent = "0";
}

let ragPollTimer = null;

function renderRagStatus(status) {
  const el = document.getElementById("rag-status");
  const pullBtn = document.getElementById("rag-pull-now");
  if (!status) {
    el.textContent = "Status unavailable.";
    return;
  }
  const parts = [`${status.indexedMessageCount} message(s) indexed`, `${status.chunkCount} chunk(s)`];
  if (status.running) {
    parts.push(`indexing in progress... ${status.processedMessages}/${status.totalMessages || "?"}`
      + (status.currentFolder ? ` (${status.currentFolder})` : ""));
  } else if (status.lastPullAt) {
    parts.push(`last pull: ${new Date(status.lastPullAt).toLocaleString()}`);
  } else {
    parts.push("never pulled");
  }
  if (status.lastError) parts.push(`last error: ${status.lastError}`);
  el.textContent = parts.join(" -- ");
  pullBtn.disabled = status.running;
}

async function refreshRagStatus() {
  const resp = await send("ragGetStatus");
  if (!resp.ok) {
    renderRagStatus(null);
    return;
  }
  renderRagStatus(resp.data);

  if (resp.data.running && !ragPollTimer) {
    ragPollTimer = setInterval(async () => {
      const poll = await send("ragGetStatus");
      if (poll.ok) renderRagStatus(poll.data);
      if (!poll.ok || !poll.data.running) {
        clearInterval(ragPollTimer);
        ragPollTimer = null;
      }
    }, 1500);
  }
}

async function pullNow() {
  await send("ragPullNow");
  await refreshRagStatus();
}

async function clearIndex() {
  await send("ragClearIndex");
  await refreshRagStatus();
}

document.getElementById("save").addEventListener("click", save);
document.getElementById("clear-history").addEventListener("click", clearHistory);
document.getElementById("rag-pull-now").addEventListener("click", pullNow);
document.getElementById("rag-clear-index").addEventListener("click", clearIndex);

load();
