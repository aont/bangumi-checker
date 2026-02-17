const byId = (id) => document.getElementById(id);

let scriptEditor;

function getScriptContent() {
  return scriptEditor ? scriptEditor.getValue() : byId("script-content").value;
}

function setScriptContent(content) {
  if (scriptEditor) {
    scriptEditor.setValue(content);
    scriptEditor.refresh();
    return;
  }
  byId("script-content").value = content;
}

function initScriptEditor() {
  if (typeof CodeMirror === "undefined") {
    return;
  }

  scriptEditor = CodeMirror.fromTextArea(byId("script-content"), {
    mode: "python",
    lineNumbers: true,
    theme: "default",
    indentUnit: 4,
    tabSize: 4,
    lineWrapping: false,
  });

  scriptEditor.setSize("100%", "340px");
}

const BACKEND_URL_STORAGE_KEY = "bangumi-checker.backend-base-url";

function normalizeBackendBaseUrl(rawValue) {
  const value = (rawValue || "").trim();
  if (!value) {
    return "";
  }

  let parsed;
  try {
    parsed = new URL(value);
  } catch (_error) {
    throw new Error("Backend URL must be an absolute URL (e.g. http://127.0.0.1:8080)");
  }

  if (!["http:", "https:"].includes(parsed.protocol)) {
    throw new Error("Backend URL must start with http:// or https://");
  }

  return parsed.origin;
}

function getBackendBaseUrl() {
  return localStorage.getItem(BACKEND_URL_STORAGE_KEY) || "";
}

function setBackendBaseUrl(value) {
  if (value) {
    localStorage.setItem(BACKEND_URL_STORAGE_KEY, value);
  } else {
    localStorage.removeItem(BACKEND_URL_STORAGE_KEY);
  }
}

function buildApiUrl(path) {
  const baseUrl = getBackendBaseUrl();
  if (!baseUrl) {
    return path;
  }
  return `${baseUrl}${path}`;
}

async function request(path, options = {}) {
  const response = await fetch(buildApiUrl(path), {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const message = data?.error || text || response.statusText;
    throw new Error(message);
  }
  return data;
}

function fillGrid(container, data) {
  container.innerHTML = "";
  Object.entries(data).forEach(([key, value]) => {
    const item = document.createElement("div");
    item.innerHTML = `<strong>${key}</strong><br>${Array.isArray(value) ? value.join(",") : String(value)}`;
    container.appendChild(item);
  });
}

function updateBackendStatus(message) {
  byId("backend-status").textContent = message;
}

function syncBackendForm() {
  const form = byId("backend-form");
  form.backend_url.value = getBackendBaseUrl();
  const baseUrl = getBackendBaseUrl();
  updateBackendStatus(baseUrl ? `Using backend: ${baseUrl}` : "Using current origin as backend");
}

async function applyBackendUrl(event) {
  event.preventDefault();
  const form = event.currentTarget;
  try {
    const normalized = normalizeBackendBaseUrl(form.backend_url.value);
    setBackendBaseUrl(normalized);
    syncBackendForm();
    await Promise.all([loadStatus(), loadConfig(), loadScript(), loadEvents(false)]);
  } catch (error) {
    updateBackendStatus(`Backend URL error: ${error.message}`);
  }
}

async function resetBackendUrl() {
  setBackendBaseUrl("");
  syncBackendForm();
  await Promise.all([loadStatus(), loadConfig(), loadScript(), loadEvents(false)]);
}

async function loadStatus() {
  const status = await request("/api/status");
  fillGrid(byId("status-grid"), status);
}

async function loadConfig() {
  const config = await request("/api/config");
  const form = byId("config-form");
  form.timeout.value = config.timeout;
  form.interval_hours.value = config.interval_hours;
  form.ggm_group_ids.value = (config.ggm_group_ids || []).join(",");
  form.code_path.value = config.code_path || "";
  form.enabled.checked = !!config.enabled;
}

async function saveConfig(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const payload = {
    timeout: Number(form.timeout.value),
    interval_hours: Number(form.interval_hours.value),
    ggm_group_ids: form.ggm_group_ids.value.split(",").map((v) => Number(v.trim())).filter(Boolean),
    code_path: form.code_path.value,
    enabled: form.enabled.checked,
  };
  await request("/api/config", { method: "PATCH", body: JSON.stringify(payload) });
  await loadStatus();
}

async function loadScript() {
  const data = await request("/api/script");
  setScriptContent(data.content || "");
  byId("script-status").textContent = "Loaded script";
}

async function saveScript() {
  const content = getScriptContent();
  await request("/api/script", { method: "PUT", body: JSON.stringify({ content }) });
  byId("script-status").textContent = "Saved script";
}

async function validateScript() {
  const content = getScriptContent();
  const result = await request("/api/script/validate", { method: "POST", body: JSON.stringify({ content }) });
  byId("script-status").textContent = result.ok ? "Validation OK" : "Validation failed";
}

async function runAction(action) {
  const body = action === "fetch-details" ? { limit: 1 } : action === "evaluate" ? { force: false } : {};
  const data = await request(`/api/actions/${action}`, { method: "POST", body: JSON.stringify(body) });
  byId("action-status").textContent = `${action} completed: ${JSON.stringify(data)}`;
  await loadStatus();
}

function renderEvents(items) {
  const body = byId("events-body");
  body.innerHTML = "";
  items.forEach((item) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${item.id}</td>
      <td>${item.broadcast_date || ""}</td>
      <td>${item.title || item.metadata_title || ""}</td>
      <td>${item.channel_name || ""}</td>
      <td>${item.user_function_returned_true ? "Yes" : "No"}</td>
      <td>${item.need_detail_fetch ? "Yes" : "No"}</td>
    `;
    body.appendChild(tr);
  });
}

async function loadEvents(matchesOnly = false) {
  const endpoint = matchesOnly ? "/api/matches?limit=30&offset=0" : "/api/events?limit=30&offset=0";
  const data = await request(endpoint);
  renderEvents(data.items || []);
}

async function init() {
  byId("backend-form").addEventListener("submit", applyBackendUrl);
  byId("reset-backend-url").addEventListener("click", resetBackendUrl);
  byId("refresh-status").addEventListener("click", loadStatus);
  byId("config-form").addEventListener("submit", saveConfig);
  byId("load-script").addEventListener("click", loadScript);
  byId("save-script").addEventListener("click", saveScript);
  byId("validate-script").addEventListener("click", validateScript);
  byId("load-events").addEventListener("click", () => loadEvents(false));

  initScriptEditor();
  byId("load-matches").addEventListener("click", () => loadEvents(true));

  document.querySelectorAll("button[data-action]").forEach((button) => {
    button.addEventListener("click", () => runAction(button.dataset.action));
  });

  syncBackendForm();
  await Promise.all([loadStatus(), loadConfig(), loadScript(), loadEvents(false)]);
}

init().catch((error) => {
  byId("action-status").textContent = `Initialization failed: ${error.message}`;
});
