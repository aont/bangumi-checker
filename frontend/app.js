const byId = (id) => document.getElementById(id);

async function request(path, options = {}) {
  const response = await fetch(path, {
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
  byId("script-content").value = data.content || "";
  byId("script-status").textContent = "Loaded script";
}

async function saveScript() {
  const content = byId("script-content").value;
  await request("/api/script", { method: "PUT", body: JSON.stringify({ content }) });
  byId("script-status").textContent = "Saved script";
}

async function validateScript() {
  const content = byId("script-content").value;
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
  byId("refresh-status").addEventListener("click", loadStatus);
  byId("config-form").addEventListener("submit", saveConfig);
  byId("load-script").addEventListener("click", loadScript);
  byId("save-script").addEventListener("click", saveScript);
  byId("validate-script").addEventListener("click", validateScript);
  byId("load-events").addEventListener("click", () => loadEvents(false));
  byId("load-matches").addEventListener("click", () => loadEvents(true));

  document.querySelectorAll("button[data-action]").forEach((button) => {
    button.addEventListener("click", () => runAction(button.dataset.action));
  });

  await Promise.all([loadStatus(), loadConfig(), loadScript(), loadEvents(false)]);
}

init().catch((error) => {
  byId("action-status").textContent = `Initialization failed: ${error.message}`;
});
