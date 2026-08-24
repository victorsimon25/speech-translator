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

const PIP_MAX = 3;

const PIP_CSS = `
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  width: 100%; height: 100%;
  background: rgba(10, 12, 18, 0.88);
  backdrop-filter: blur(28px);
  -webkit-backdrop-filter: blur(28px);
  font-family: system-ui, -apple-system, sans-serif;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.accent-bar {
  height: 3px;
  background: linear-gradient(90deg, #40c878 0%, #5598e8 100%);
  flex-shrink: 0;
}
#captions {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  padding: 8px 18px 12px;
  overflow: hidden;
}
.caption-card {
  padding: 3px 0 5px;
  transition: opacity 0.5s ease;
  animation: slideUp 0.28s cubic-bezier(0.22, 1, 0.36, 1);
}
.caption-card.age-0 { opacity: 1; }
.caption-card.age-1 { opacity: 0.52; }
.caption-card.age-2 { opacity: 0.22; }
@keyframes slideUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
.target-text {
  font-size: 22px;
  font-weight: 500;
  color: #fff;
  text-shadow: 0 1px 8px rgba(0,0,0,0.95), 0 0 2px rgba(0,0,0,0.6);
  line-height: 1.3;
  letter-spacing: 0.01em;
}
.source-text {
  font-size: 13px;
  color: rgba(255,255,255,0.42);
  margin-top: 2px;
  line-height: 1.3;
}
`;

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
  ta: "TA", ml: "ML", kn: "KN",
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
let pipDiv = null;

const bc = new BroadcastChannel("speech-translator");

// ── WebSocket ─────────────────────────────────────────────────────────

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    handleMessage(msg);
    bc.postMessage(msg);
    if (pipDiv && msg.type === "caption") renderIntoPip(msg);
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

// ── PiP caption renderer ──────────────────────────────────────────────

function renderIntoPip(msg) {
  if (!pipDiv) return;
  const doc = pipDiv.ownerDocument;
  const card = doc.createElement("div");
  card.className = "caption-card";
  if (msg.target_text !== null && msg.target_text !== undefined) {
    const tgt = doc.createElement("div");
    tgt.className = "target-text";
    tgt.textContent = msg.target_text;
    card.appendChild(tgt);
    if (msg.source_text && msg.source_text !== msg.target_text) {
      const src = doc.createElement("div");
      src.className = "source-text";
      src.textContent = msg.source_text;
      card.appendChild(src);
    }
  } else {
    const tgt = doc.createElement("div");
    tgt.className = "target-text";
    tgt.textContent = msg.source_text || "(no text)";
    card.appendChild(tgt);
  }
  pipDiv.appendChild(card);
  while (pipDiv.children.length > PIP_MAX) pipDiv.removeChild(pipDiv.firstChild);
  const cards = pipDiv.querySelectorAll(".caption-card");
  const total = cards.length;
  cards.forEach((c, i) => {
    c.classList.remove("age-0", "age-1", "age-2");
    c.classList.add(`age-${total - 1 - i}`);
  });
}

startBtn.addEventListener("click", () => {
  send({ type: "start" });
});

stopBtn.addEventListener("click", () => {
  send({ type: "stop" });
});

document.getElementById("subtitle-btn").addEventListener("click", async () => {
  if (window.documentPictureInPicture) {
    try {
      const pipWin = await window.documentPictureInPicture.requestWindow({
        width: 860, height: 155,
      });
      const style = pipWin.document.createElement("style");
      style.textContent = PIP_CSS;
      pipWin.document.head.appendChild(style);
      const bar = pipWin.document.createElement("div");
      bar.className = "accent-bar";
      pipWin.document.body.appendChild(bar);
      pipDiv = pipWin.document.createElement("div");
      pipDiv.id = "captions";
      pipWin.document.body.appendChild(pipDiv);
      pipWin.addEventListener("pagehide", () => { pipDiv = null; });
    } catch {
      window.open("/subtitle", "_blank");
    }
  } else {
    window.open("/subtitle", "_blank");
  }
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
