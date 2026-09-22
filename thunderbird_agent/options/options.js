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
  document.getElementById("history-count").textContent = (resp.data.history || []).length;
}

async function save() {
  const settings = {};
  for (const key of fields) {
    settings[key] = document.getElementById(key).value.trim();
  }
  await send("saveSettings", { settings });

  const statusEl = document.getElementById("save-status");
  statusEl.hidden = false;
  statusEl.textContent = "Saved.";
  setTimeout(() => { statusEl.hidden = true; }, 2000);
}

async function clearHistory() {
  await send("saveSettings", { settings: { history: [] } });
  document.getElementById("history-count").textContent = "0";
}

document.getElementById("save").addEventListener("click", save);
document.getElementById("clear-history").addEventListener("click", clearHistory);

load();
