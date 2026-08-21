/**
 * Speech Translator — vanilla JS WebSocket client (D38).
 *
 * State machine mirrors the server: idle | loading | listening | running |
 * degraded | error.  Caption cards are append-only (D11).
 */

"use strict";

// ── Config ────────────────────────────────────────────────────────────

const SILENCE_THRESHOLD_MS = 600;       // matches config.silence_threshold_ms (D25)
const GAP_THRESHOLD_MS = SILENCE_THRESHOLD_MS * 2;  // show divider above this

const STATE_LABELS = {
  idle:      "Idle",
  loading:   "Loading…",
  listening: "Listening",
  running:   "Running",
  degraded:  "Degraded",
  error:     "Error",
};

const LANG_NAMES = {
  en: "EN", es: "ES", fr: "FR", de: "DE", it: "IT", pt: "PT",
  zh: "ZH", ja: "JA", ko: "KO", ru: "RU", ar: "AR", hi: "HI",
};

// ── DOM refs ──────────────────────────────────────────────────────────

const statusBadge    = document.getElementById("status-badge");
const statusDot      = statusBadge.querySelector(".dot");
const statusLabel    = statusBadge.querySelector(".label");
const speakingDot    = document.getElementById("speaking-dot");
const srcLangChip    = document.getElementById("src-lang-chip");
const tgtLangSelect  = document.getElementById("tgt-lang-select");
const healthBadge    = document.getElementById("health-badge");
const captionsWrapper = document.getElementById("captions-wrapper");
const captionsDiv    = document.getElementById("captions");
const jumpBtn        = document.getElementById("jump-btn");
const startBtn       = document.getElementById("start-btn");
const stopBtn        = document.getElementById("stop-btn");
const exportBtn      = document.getElementById("export-btn");
const stateDetail    = document.getElementById("state-detail");

// ── App state ─────────────────────────────────────────────────────────

let ws = null;
let currentState = "idle";
let detectedSrcLang = null;
let overrideActive = false;
let isAtBottom = true;        // auto-scroll tracking

// ── WebSocket ─────────────────────────────────────────────────────────

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    handleMessage(msg);
  };

  ws.onclose = () => {
    ws = null;
    setUiState("idle");
    setTimeout(connect, 2000);   // reconnect
  };

  ws.onerror = () => ws.close();
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

// ── Message dispatch ──────────────────────────────────────────────────

function handleMessage(msg) {
  switch (msg.type) {
    case "state":
      setUiState(msg.state, msg.detail || "");
      break;
    case "caption":
      appendCaption(msg);
      break;
    case "vad":
      speakingDot.classList.toggle("speaking", !!msg.speaking);
      break;
    case "health":
      updateHealth(msg);
      break;
    case "language_detected":
      onLanguageDetected(msg.lang, msg.confidence);
      break;
    case "dropped":
      appendDroppedNote(msg);
      break;
    default:
      break;
  }
}

// ── State UI ──────────────────────────────────────────────────────────

function setUiState(state, detail) {
  currentState = state;

  // Remove all state classes, add the new one.
  statusBadge.className = `state-${state}`;
  statusLabel.textContent = STATE_LABELS[state] || state;
  stateDetail.textContent = detail || "";

  startBtn.disabled = state !== "idle";
  stopBtn.disabled  = state === "idle" || state === "error";

  if (state === "idle" || state === "error") {
    speakingDot.classList.remove("speaking");
  }
}

// ── Caption cards ─────────────────────────────────────────────────────

function appendCaption(msg) {
  // Gap divider when there was a long silence before this utterance.
  if ((msg.preceding_silence_ms || 0) >= GAP_THRESHOLD_MS) {
    const gap = document.createElement("div");
    gap.className = "gap-divider";
    const secs = ((msg.preceding_silence_ms || 0) / 1000).toFixed(1);
    gap.textContent = `${secs}s pause`;
    captionsDiv.appendChild(gap);
  }

  const card = document.createElement("div");
  card.className = "caption-card";
  card.dataset.id = msg.id;

  if (msg.target_text !== null && msg.target_text !== undefined) {
    const tgt = document.createElement("div");
    tgt.className = "target-text";
    tgt.textContent = msg.target_text;
    card.appendChild(tgt);

    if (msg.source_text && msg.source_text !== msg.target_text) {
      const src = document.createElement("div");
      src.className = "source-text";
      src.textContent = msg.source_text;
      card.appendChild(src);
    }
  } else {
    // Translation unavailable — show source text prominently.
    card.classList.add("no-translation");
    const src = document.createElement("div");
    src.className = "target-text";
    src.textContent = msg.source_text || "(no text)";
    card.appendChild(src);

    const note = document.createElement("div");
    note.className = "unavailable-note";
    note.textContent = msg.mt_error
      ? `Translation unavailable: ${msg.mt_error}`
      : "Translation unavailable";
    card.appendChild(note);
  }

  captionsDiv.appendChild(card);
  maybeScrollToBottom();
}

function appendDroppedNote(msg) {
  const note = document.createElement("div");
  note.className = "gap-divider";
  note.textContent = `⚠ ${msg.utterances} utterance(s) dropped (backpressure)`;
  captionsDiv.appendChild(note);
  maybeScrollToBottom();
}

// ── Auto-scroll ───────────────────────────────────────────────────────

function maybeScrollToBottom() {
  if (isAtBottom) {
    captionsWrapper.scrollTop = captionsWrapper.scrollHeight;
  }
}

captionsWrapper.addEventListener("scroll", () => {
  const threshold = 40;
  const atBottom =
    captionsWrapper.scrollHeight - captionsWrapper.scrollTop - captionsWrapper.clientHeight < threshold;
  isAtBottom = atBottom;
  jumpBtn.classList.toggle("visible", !atBottom);
});

jumpBtn.addEventListener("click", () => {
  captionsWrapper.scrollTop = captionsWrapper.scrollHeight;
  isAtBottom = true;
  jumpBtn.classList.remove("visible");
});

// ── Health ────────────────────────────────────────────────────────────

function updateHealth(msg) {
  const rtfStr = typeof msg.rtf === "number" ? msg.rtf.toFixed(2) : "–";
  const ageStr = typeof msg.word_age_ms === "number"
    ? `${(msg.word_age_ms / 1000).toFixed(1)}s` : "–";
  const qStr   = typeof msg.queue_depth === "number" ? `q${msg.queue_depth}` : "";
  const budgetPct = msg.chars_budget > 0
    ? Math.round(100 * msg.chars_used / msg.chars_budget) : 0;

  healthBadge.textContent = `RTF ${rtfStr}  age ${ageStr}  ${qStr}  budget ${budgetPct}%`;
}

// ── Language detection ────────────────────────────────────────────────

function onLanguageDetected(lang, confidence) {
  if (overrideActive) return;
  detectedSrcLang = lang;
  const label = LANG_NAMES[lang] || lang.toUpperCase();
  srcLangChip.textContent = `${label} (detected)`;
  srcLangChip.classList.add("detected");
  srcLangChip.title = `Detected ${lang} (${(confidence * 100).toFixed(0)}% confidence) — click to override`;
}

srcLangChip.addEventListener("click", () => {
  const lang = prompt("Override source language (ISO 639-1, e.g. 'es'):", detectedSrcLang || "");
  if (lang && /^[a-z]{2,3}$/.test(lang.trim())) {
    const l = lang.trim();
    overrideActive = true;
    srcLangChip.textContent = `${LANG_NAMES[l] || l.toUpperCase()} (locked)`;
    srcLangChip.classList.remove("detected");
    send({ type: "override_source_lang", lang: l });
  }
});

tgtLangSelect.addEventListener("change", () => {
  send({ type: "set_target_lang", lang: tgtLangSelect.value });
});

// ── Controls ──────────────────────────────────────────────────────────

startBtn.addEventListener("click", () => {
  send({ type: "start" });
});

stopBtn.addEventListener("click", () => {
  send({ type: "stop" });
});

exportBtn.addEventListener("click", () => {
  const cards = captionsDiv.querySelectorAll(".caption-card");
  const lines = [];
  cards.forEach(card => {
    const tgt = card.querySelector(".target-text");
    const src = card.querySelector(".source-text");
    if (tgt) lines.push(tgt.textContent);
    if (src) lines.push(`  [${src.textContent}]`);
    lines.push("");
  });
  const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `transcript-${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

// ── Init ──────────────────────────────────────────────────────────────

setUiState("idle", "");
connect();
