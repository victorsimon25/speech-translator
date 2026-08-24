"use strict";

const MAX = 3;
const captionsDiv = document.getElementById("captions");
let pipDiv = null;

function renderInto(container, msg) {
  // Remove empty-hint on first real caption
  const hint = container.querySelector(".empty-hint");
  if (hint) hint.remove();

  const doc = container.ownerDocument;
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

  container.appendChild(card);

  while (container.children.length > MAX) {
    container.removeChild(container.firstChild);
  }

  const cards = container.querySelectorAll(".caption-card");
  const total = cards.length;
  cards.forEach((c, i) => {
    c.classList.remove("age-0", "age-1", "age-2");
    c.classList.add(`age-${total - 1 - i}`);
  });
}

function appendCaption(msg) {
  renderInto(captionsDiv, msg);
  if (pipDiv) renderInto(pipDiv, msg);
}

// ── Float (PiP) button ────────────────────────────────────────────────

document.getElementById("pip-btn").addEventListener("click", async () => {
  if (!window.documentPictureInPicture) {
    alert("Document Picture-in-Picture requires Chrome or Edge 116+.");
    return;
  }
  const pipWin = await window.documentPictureInPicture.requestWindow({
    width: 860, height: 155,
  });
  const link = pipWin.document.createElement("link");
  link.rel = "stylesheet";
  link.href = "/static/subtitle.css";
  pipWin.document.head.appendChild(link);
  const bar = pipWin.document.createElement("div");
  bar.className = "accent-bar";
  pipWin.document.body.appendChild(bar);
  pipDiv = pipWin.document.createElement("div");
  pipDiv.id = "captions";
  pipWin.document.body.appendChild(pipDiv);
  pipWin.addEventListener("pagehide", () => { pipDiv = null; });
});

// ── BroadcastChannel ──────────────────────────────────────────────────

const bc = new BroadcastChannel("speech-translator");
bc.onmessage = ({ data: msg }) => {
  if (msg.type === "caption") appendCaption(msg);
};
