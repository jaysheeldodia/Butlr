const CARDS_PER_PAGE = 3;
const COUNTDOWN_CIRCUMFERENCE = 2 * Math.PI * 20;

const LANE_CONFIG = {
  high: {
    title: "High Priority",
    empty: "No high-priority email tasks.",
  },
  medium: {
    title: "Needs Attention",
    empty: "Nothing waiting for attention.",
  },
  low: {
    title: "Other",
    empty: "No additional email tasks.",
  },
};

const state = {
  payload: null,
  tasks: [],
  drafts: {},
  lanePage: {
    high: 0,
    medium: 0,
    low: 0,
  },
  laneDirection: {
    high: null,
    medium: null,
    low: null,
  },
  pollIntervalSeconds: 0,
  nextPollAt: null,
  lastRunAt: null,       // Last known last_run_at value for dedup
  stallRetryCount: 0,    // Consecutive fetches with no new data
  refreshTimeoutId: null,
  countdownIntervalId: null,
  isLoading: false,
};

const elements = {
  app: document.querySelector("#app"),
  authPrompt: document.querySelector("#authPrompt"),
  emptyState: document.querySelector("#emptyState"),
  lanes: document.querySelector("#lanes"),
  statusBanner: document.querySelector("#statusBanner"),
  disconnectButton: document.querySelector("#disconnectButton"),
  runNowButton: document.querySelector("#runNowButton"),
  countdownProgress: document.querySelector("#countdownProgress"),
  countdownValue: document.querySelector("#countdownValue"),
  countdownText: document.querySelector("#countdownText"),
  lastRunLabel: document.querySelector("#lastRunLabel"),
  taskDialog: document.querySelector("#taskDialog"),
  closeDialogButton: document.querySelector("#closeDialogButton"),
  dialogCloseAction: document.querySelector("#dialogCloseAction"),
  copyDraftButton: document.querySelector("#copyDraftButton"),
  dialogTitle: document.querySelector("#dialogTitle"),
  dialogPriority: document.querySelector("#dialogPriority"),
  dialogSource: document.querySelector("#dialogSource"),
  dialogDeadline: document.querySelector("#dialogDeadline"),
  dialogSummary: document.querySelector("#dialogSummary"),
  dialogDraftSubject: document.querySelector("#dialogDraftSubject"),
  dialogDraft: document.querySelector("#dialogDraft"),
  draftSection: document.querySelector("#draftSection"),
  laneTemplate: document.querySelector("#laneTemplate"),
  cardTemplate: document.querySelector("#cardTemplate"),
};

// ── Utilities ─────────────────────────────────────────────────────────────────

function normalizeDeadline(value) {
  if (value == null) return null;
  const cleaned = String(value).trim();
  if (!cleaned || ["null", "none", "n/a"].includes(cleaned.toLowerCase())) return null;
  return cleaned;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function truncate(text, maxLength) {
  return text.length <= maxLength ? text : `${text.slice(0, maxLength - 3).trimEnd()}...`;
}

function timeAgo(isoString) {
  if (!isoString) return null;
  const diffMs = Date.now() - Date.parse(isoString);
  if (Number.isNaN(diffMs) || diffMs < 0) return null;
  const diffSeconds = Math.floor(diffMs / 1000);
  if (diffSeconds < 60) return "just now";
  const diffMinutes = Math.floor(diffSeconds / 60);
  if (diffMinutes < 60) return `${diffMinutes}m ago`;
  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  return `${Math.floor(diffHours / 24)}d ago`;
}

// ── Status banner ──────────────────────────────────────────────────────────────

function setStatus(message, tone = "info") {
  if (!message) {
    elements.statusBanner.hidden = true;
    elements.statusBanner.textContent = "";
    elements.statusBanner.removeAttribute("data-tone");
    return;
  }

  elements.statusBanner.hidden = false;
  elements.statusBanner.dataset.tone = tone;
  elements.statusBanner.textContent = message;
}

// ── Countdown ──────────────────────────────────────────────────────────────────

function formatSeconds(totalSeconds) {
  const safeSeconds = Math.max(0, Math.ceil(totalSeconds));
  const minutes = Math.floor(safeSeconds / 60);
  const seconds = safeSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function clearRefreshTimeout() {
  if (state.refreshTimeoutId) {
    window.clearTimeout(state.refreshTimeoutId);
    state.refreshTimeoutId = null;
  }
}

function clearCountdownInterval() {
  if (state.countdownIntervalId) {
    window.clearInterval(state.countdownIntervalId);
    state.countdownIntervalId = null;
  }
}

function setCountdownState(secondsRemaining, progress) {
  elements.countdownValue.textContent =
    secondsRemaining > 0 ? formatSeconds(secondsRemaining) : "0:00";
  elements.countdownText.textContent =
    secondsRemaining > 0 ? `Updates in ${formatSeconds(secondsRemaining)}` : "Checking now…";
  elements.countdownProgress.style.strokeDasharray = String(COUNTDOWN_CIRCUMFERENCE);
  elements.countdownProgress.style.strokeDashoffset = String(
    COUNTDOWN_CIRCUMFERENCE * Math.min(Math.max(progress, 0), 1),
  );
}

function startCountdown() {
  clearCountdownInterval();

  if (!state.nextPollAt) {
    elements.countdownValue.textContent = "--";
    elements.countdownText.textContent = "Waiting for schedule";
    elements.countdownProgress.style.strokeDashoffset = String(COUNTDOWN_CIRCUMFERENCE);
    return;
  }

  const nextPollAtMs = Date.parse(state.nextPollAt);
  if (Number.isNaN(nextPollAtMs) || !state.pollIntervalSeconds) {
    elements.countdownValue.textContent = "--";
    elements.countdownText.textContent = "Waiting for schedule";
    elements.countdownProgress.style.strokeDashoffset = String(COUNTDOWN_CIRCUMFERENCE);
    return;
  }

  const update = () => {
    const remainingSeconds = (nextPollAtMs - Date.now()) / 1000;
    const elapsedSeconds = state.pollIntervalSeconds - remainingSeconds;
    const progress = elapsedSeconds / state.pollIntervalSeconds;
    setCountdownState(remainingSeconds, progress);
  };

  update();
  state.countdownIntervalId = window.setInterval(update, 1000);
}

function updateLastRunLabel() {
  if (!elements.lastRunLabel) return;
  const ago = timeAgo(state.lastRunAt);
  elements.lastRunLabel.textContent = ago ? `Last run: ${ago}` : "";
  elements.lastRunLabel.hidden = !ago;
}

// ── Auto-refresh scheduling ────────────────────────────────────────────────────

function scheduleAutoRefresh() {
  clearRefreshTimeout();

  if (!state.nextPollAt) {
    // No schedule known — retry with exponential backoff
    const delay = Math.min(5000 * Math.pow(2, state.stallRetryCount), 30_000);
    state.refreshTimeoutId = window.setTimeout(() => loadDashboard(), delay);
    return;
  }

  const nextPollAtMs = Date.parse(state.nextPollAt);
  if (Number.isNaN(nextPollAtMs)) {
    state.refreshTimeoutId = window.setTimeout(() => loadDashboard(), 5000);
    return;
  }

  const delayMs = nextPollAtMs - Date.now();

  if (delayMs <= 0 && state.stallRetryCount > 0) {
    // Backend hasn't updated yet — back off to avoid hammering it
    const backoff = Math.min(3000 * state.stallRetryCount, 30_000);
    state.refreshTimeoutId = window.setTimeout(() => loadDashboard(), backoff);
    return;
  }

  // Normal case: fire slightly after the expected poll time
  state.refreshTimeoutId = window.setTimeout(
    () => loadDashboard(),
    Math.max(delayMs + 500, 1000),
  );
}

// ── Task grouping & rendering ──────────────────────────────────────────────────

function groupTasks(tasks) {
  return tasks.reduce(
    (groups, task) => {
      const priority = task.priority in groups ? task.priority : "low";
      groups[priority].push(task);
      return groups;
    },
    { high: [], medium: [], low: [] },
  );
}

function getTaskFromQuery() {
  return new URLSearchParams(window.location.search).get("task");
}

function updateQuery(taskId) {
  const url = new URL(window.location.href);
  if (taskId) {
    url.searchParams.set("task", taskId);
  } else {
    url.searchParams.delete("task");
  }
  window.history.replaceState({}, "", url);
}

function renderCard(task) {
  const fragment = elements.cardTemplate.content.cloneNode(true);
  const button = fragment.querySelector(".task-card");
  const title = fragment.querySelector(".task-title");
  const summary = fragment.querySelector(".task-summary");
  const source = fragment.querySelector(".task-source");
  const priorityChip = fragment.querySelector(".task-priority-chip");

  title.textContent = task.title;
  summary.textContent = truncate(task.description, 110);
  source.textContent = task.source === "calendar" ? "Calendar" : "Email";

  if (priorityChip) {
    const labels = { high: "High", medium: "Medium", low: "Low" };
    priorityChip.textContent = labels[task.priority] ?? task.priority;
    priorityChip.dataset.priority = task.priority;
  }

  button.addEventListener("click", () => openTaskDialog(task.id));
  button.setAttribute("aria-label", `Open details for ${escapeHtml(task.title)}`);
  button.dataset.priority = task.priority;
  return fragment;
}

function renderLane(priority, tasks) {
  const config = LANE_CONFIG[priority];
  const fragment = elements.laneTemplate.content.cloneNode(true);
  const lane = fragment.querySelector(".lane");
  const title = fragment.querySelector(".lane-title");
  const caption = fragment.querySelector(".lane-caption");
  const prev = fragment.querySelector(".lane-prev");
  const next = fragment.querySelector(".lane-next");
  const cards = fragment.querySelector(".lane-cards");

  title.textContent = config.title;

  if (!tasks.length) {
    caption.textContent = config.empty;
    prev.disabled = true;
    next.disabled = true;
    cards.innerHTML = `<div class="empty-state"><p>${escapeHtml(config.empty)}</p></div>`;
    cards.style.gridTemplateColumns = "1fr";
    lane.dataset.priority = priority;
    return fragment;
  }

  const totalPages = Math.max(Math.ceil(tasks.length / CARDS_PER_PAGE), 1);
  const page = Math.min(state.lanePage[priority], totalPages - 1);
  const start = page * CARDS_PER_PAGE;
  const pageTasks = tasks.slice(start, start + CARDS_PER_PAGE);

  caption.textContent = `${tasks.length} task${tasks.length !== 1 ? "s" : ""} · page ${page + 1} of ${totalPages}`;
  prev.disabled = page === 0;
  next.disabled = page >= totalPages - 1;

  prev.addEventListener("click", () => {
    rerenderLane(priority, Math.max(0, page - 1), "back");
  });

  next.addEventListener("click", () => {
    rerenderLane(priority, Math.min(totalPages - 1, page + 1), "forward");
  });

  const direction = state.laneDirection[priority];
  if (direction) {
    cards.dataset.animate = direction;
    state.laneDirection[priority] = null;
  }

  for (const task of pageTasks) {
    cards.appendChild(renderCard(task));
  }

  lane.dataset.priority = priority;
  return fragment;
}

function renderDialog(task) {
  elements.dialogTitle.textContent = task.title;
  const priorityLabel = `${task.priority.replaceAll("_", " ")} priority`;
  elements.dialogPriority.textContent = priorityLabel;
  elements.dialogPriority.dataset.priority = task.priority;
  elements.dialogSource.textContent = task.source === "calendar" ? "Calendar" : "Email";
  elements.dialogSummary.textContent = task.description;

  const deadline = normalizeDeadline(task.deadline);
  elements.dialogDeadline.hidden = !deadline;
  if (deadline) {
    elements.dialogDeadline.textContent = `Due: ${deadline}`;
  } else {
    elements.dialogDeadline.textContent = "";
  }

  const draft = task.related_email_id ? state.drafts[task.related_email_id] : null;
  if (draft) {
    elements.draftSection.hidden = false;
    elements.dialogDraftSubject.textContent = `Subject: ${draft.subject}`;
    elements.dialogDraft.textContent = draft.body;
  } else {
    elements.draftSection.hidden = true;
    elements.dialogDraftSubject.textContent = "";
    elements.dialogDraft.textContent = "";
  }
}

function openTaskDialog(taskId) {
  const task = state.tasks.find((item) => item.id === taskId);
  if (!task) return;

  renderDialog(task);
  updateQuery(taskId);
  if (!elements.taskDialog.open) {
    elements.taskDialog.showModal();
  }
}

function closeTaskDialog() {
  updateQuery(null);
  if (elements.taskDialog.open) {
    elements.taskDialog.close();
  }
}

async function copyDraftToClipboard() {
  const subject = elements.dialogDraftSubject.textContent;
  const body = elements.dialogDraft.textContent;
  if (!body) return;

  const text = subject ? `${subject}\n\n${body}` : body;
  try {
    await navigator.clipboard.writeText(text);
    setStatus("Suggested reply copied to clipboard.");
    window.setTimeout(() => setStatus(""), 3000);
  } catch (_error) {
    setStatus("Could not copy the suggested reply.", "error");
  }
}

function rerenderLane(priority, newPage, direction) {
  state.laneDirection[priority] = direction;
  state.lanePage[priority] = newPage;

  const existingLane = elements.lanes.querySelector(`.lane[data-priority="${priority}"]`);
  if (!existingLane) {
    renderApp();
    return;
  }

  const grouped = groupTasks(state.tasks);
  const fragment = renderLane(priority, grouped[priority]);
  existingLane.replaceWith(fragment);
}

// ── Main render ────────────────────────────────────────────────────────────────

function renderApp() {
  const payload = state.payload ?? {};
  const result = payload.result;
  state.pollIntervalSeconds = Number(payload.poll_interval_seconds ?? 0);
  state.nextPollAt = payload.next_poll_at ?? null;

  if (payload.last_error) {
    setStatus(payload.last_error, "error");
  } else if (!payload.last_run_at) {
    setStatus("Waiting for the backend poller to complete its first run.", "info");
  } else {
    setStatus("");
  }

  updateLastRunLabel();
  startCountdown();
  scheduleAutoRefresh();

  if (!result || !Array.isArray(result.tasks) || result.tasks.length === 0) {
    elements.app.hidden = false;
    elements.emptyState.hidden = false;
    elements.emptyState.innerHTML = `
      <div class="empty-state">
        <div>
          <p class="eyebrow">No Tasks</p>
          <h2>Nothing actionable right now</h2>
          <p class="subtitle">No email tasks were extracted from the latest run.</p>
        </div>
      </div>
    `;
    elements.lanes.replaceChildren();
    state.tasks = [];
    state.drafts = {};
    closeTaskDialog();
    return;
  }

  elements.emptyState.hidden = true;
  elements.app.hidden = false;
  state.tasks = result.tasks;
  state.drafts = result.drafts ?? {};

  const groupedTasks = groupTasks(result.tasks);
  const lanes = [];

  for (const priority of ["high", "medium", "low"]) {
    lanes.push(renderLane(priority, groupedTasks[priority]));
  }

  elements.lanes.replaceChildren(...lanes);

  const selectedTaskId = getTaskFromQuery();
  if (selectedTaskId) {
    openTaskDialog(selectedTaskId);
  } else {
    closeTaskDialog();
  }
}

// ── Data fetching ──────────────────────────────────────────────────────────────

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      Accept: "application/json",
      ...(options.headers ?? {}),
    },
    ...options,
  });

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch (_error) {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }

  return response.json();
}

async function loadDashboard(path = "/result") {
  if (state.isLoading) return;

  state.isLoading = true;
  setBusy(true);

  try {
    const data = await fetchJson(path);
    const newLastRunAt = data.last_run_at ?? null;

    if (newLastRunAt !== null && newLastRunAt === state.lastRunAt) {
      // Data unchanged — update timing only, no full re-render
      state.stallRetryCount += 1;
      state.payload = data;
      state.pollIntervalSeconds = Number(data.poll_interval_seconds ?? 0);
      state.nextPollAt = data.next_poll_at ?? null;
      updateLastRunLabel();
      startCountdown();
      scheduleAutoRefresh();
    } else {
      state.stallRetryCount = 0;
      state.lastRunAt = newLastRunAt;
      state.payload = data;
      renderApp();
    }
  } catch (error) {
    setStatus(error.message || "Could not reach the API. Is the backend running?", "error");
    scheduleAutoRefresh();
  } finally {
    state.isLoading = false;
    setBusy(false);
  }
}

async function runNow() {
  if (state.isLoading) return;
  clearRefreshTimeout();

  state.isLoading = true;
  setBusy(true);

  try {
    const data = await fetchJson("/run", { method: "POST" });
    state.stallRetryCount = 0;
    state.lastRunAt = data.last_run_at ?? null;
    state.payload = data;
    renderApp();
  } catch (error) {
    setStatus(error.message || "Manual run failed.", "error");
    scheduleAutoRefresh();
  } finally {
    state.isLoading = false;
    setBusy(false);
  }
}

function setBusy(isBusy) {
  elements.disconnectButton.disabled = isBusy;
  if (elements.runNowButton) elements.runNowButton.disabled = isBusy;
}

// ── Auth ───────────────────────────────────────────────────────────────────────

async function checkAuthAndLoad() {
  try {
    const auth = await fetchJson("/auth/status");
    if (!auth.authenticated) {
      clearRefreshTimeout();
      clearCountdownInterval();
      elements.authPrompt.hidden = false;
      elements.app.hidden = true;
      elements.emptyState.hidden = true;
      elements.countdownValue.textContent = "--";
      elements.countdownText.textContent = "Waiting for schedule";
      return;
    }
  } catch (_error) {
    setStatus("Could not determine authentication status.", "error");
    elements.authPrompt.hidden = false;
    return;
  }

  elements.authPrompt.hidden = true;
  await loadDashboard("/result");
}

async function disconnect() {
  setBusy(true);
  try {
    await fetchJson("/auth/logout", { method: "POST" });
    clearRefreshTimeout();
    clearCountdownInterval();
    state.nextPollAt = null;
    state.pollIntervalSeconds = 0;
    state.lastRunAt = null;
    state.stallRetryCount = 0;
    closeTaskDialog();
    elements.app.hidden = true;
    elements.emptyState.hidden = true;
    elements.authPrompt.hidden = false;
    elements.countdownValue.textContent = "--";
    elements.countdownText.textContent = "Waiting for schedule";
    if (elements.lastRunLabel) elements.lastRunLabel.hidden = true;
    setStatus("");
  } catch (error) {
    setStatus(error.message || "Failed to disconnect Gmail.", "error");
  } finally {
    setBusy(false);
  }
}

// ── Event listeners ────────────────────────────────────────────────────────────

function initEventListeners() {
  elements.disconnectButton.addEventListener("click", disconnect);
  if (elements.runNowButton) elements.runNowButton.addEventListener("click", runNow);
  elements.closeDialogButton.addEventListener("click", closeTaskDialog);
  elements.dialogCloseAction.addEventListener("click", closeTaskDialog);
  elements.copyDraftButton.addEventListener("click", copyDraftToClipboard);
  elements.taskDialog.addEventListener("close", () => updateQuery(null));
  elements.taskDialog.addEventListener("click", (event) => {
    const bounds = elements.taskDialog.getBoundingClientRect();
    const clickedBackdrop =
      event.clientX < bounds.left ||
      event.clientX > bounds.right ||
      event.clientY < bounds.top ||
      event.clientY > bounds.bottom;

    if (clickedBackdrop) {
      closeTaskDialog();
    }
  });
  window.addEventListener("popstate", () => {
    const selectedTaskId = getTaskFromQuery();
    if (selectedTaskId) {
      openTaskDialog(selectedTaskId);
    } else {
      closeTaskDialog();
    }
  });

  const authError = new URLSearchParams(window.location.search).get("auth_error");
  if (authError) {
    setStatus(`Gmail connection failed: ${authError}`, "error");
  }
}

initEventListeners();
checkAuthAndLoad();
