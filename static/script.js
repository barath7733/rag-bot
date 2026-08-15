// ----------------------------------------------------------------------
// Frontend logic for the AI Assistant + RAG Document Intelligence app.
// No API keys or secrets ever live in this file — every privileged call
// goes through the FastAPI backend.
// ----------------------------------------------------------------------

const API = {
  chat: "/api/chat",
  upload: "/api/documents/upload",
  documents: "/api/documents",
  health: "/api/health",
  imageGenerate: "/api/image/generate",
};

const state = {
  mode: "auto",
  history: [], // {role, content}
  documents: [],
};

const el = {
  chatWindow: document.getElementById("chat-window"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  sendBtn: document.getElementById("send-btn"),
  loadingIndicator: document.getElementById("loading-indicator"),
  modeToggle: document.getElementById("mode-toggle"),
  fileInput: document.getElementById("file-input"),
  uploadStatus: document.getElementById("upload-status"),
  documentList: document.getElementById("document-list"),
  clearChatBtn: document.getElementById("clear-chat-btn"),
  healthIndicator: document.getElementById("health-indicator"),
  imageForm: document.getElementById("image-form"),
  imagePrompt: document.getElementById("image-prompt"),
  imageGenerateBtn: document.getElementById("image-generate-btn"),
};

// ---------------------------- Mode selector ----------------------------

el.modeToggle.addEventListener("click", (e) => {
  const btn = e.target.closest(".mode-btn");
  if (!btn) return;
  document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  state.mode = btn.dataset.mode;
});

// ---------------------------- Chat ----------------------------

function autoResizeTextarea() {
  el.chatInput.style.height = "auto";
  el.chatInput.style.height = Math.min(el.chatInput.scrollHeight, 160) + "px";
}
el.chatInput.addEventListener("input", autoResizeTextarea);

el.chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    el.chatForm.requestSubmit();
  }
});

function clearWelcomeMessage() {
  const welcome = el.chatWindow.querySelector(".welcome-message");
  if (welcome) welcome.remove();
}

const MODE_LABELS = {
  rag: "📄 Document Mode",
  web: "🌐 Web Search",
  general: "💬 General AI",
};

function appendMessage({ role, content, modeUsed, sources, webSources, isError }) {
  clearWelcomeMessage();
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  if (role === "assistant" && modeUsed) {
    const tag = document.createElement("div");
    tag.className = "mode-tag";
    tag.textContent = MODE_LABELS[modeUsed] || "💬 General AI";
    wrapper.appendChild(tag);
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble" + (isError ? " error-bubble" : "");
  bubble.textContent = content;
  wrapper.appendChild(bubble);

  if (sources && sources.length > 0) {
    const box = document.createElement("div");
    box.className = "sources-box";
    box.innerHTML = "<strong>Sources:</strong>";
    sources.forEach((s) => {
      const item = document.createElement("div");
      item.className = "source-item";
      const pageInfo = s.page ? `, page ${s.page}` : "";
      item.innerHTML = `<div class="source-title">${escapeHtml(s.document_name)}${pageInfo} · score ${s.score.toFixed(2)}</div><div>${escapeHtml(s.snippet)}</div>`;
      box.appendChild(item);
    });
    wrapper.appendChild(box);
  }

  if (webSources && webSources.length > 0) {
    const box = document.createElement("div");
    box.className = "sources-box";
    box.innerHTML = "<strong>Sources:</strong>";
    webSources.forEach((s) => {
      const item = document.createElement("div");
      item.className = "source-item";
      const link = document.createElement("a");
      link.href = s.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.className = "source-title";
      link.textContent = s.title;
      link.style.color = "inherit";
      item.appendChild(link);
      const snippet = document.createElement("div");
      snippet.textContent = s.snippet;
      item.appendChild(snippet);
      box.appendChild(item);
    });
    wrapper.appendChild(box);
  }

  el.chatWindow.appendChild(wrapper);
  el.chatWindow.scrollTop = el.chatWindow.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function setLoading(isLoading) {
  el.loadingIndicator.classList.toggle("hidden", !isLoading);
  el.sendBtn.disabled = isLoading;
  if (isLoading) el.chatWindow.scrollTop = el.chatWindow.scrollHeight;
}

el.chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = el.chatInput.value.trim();
  if (!question) return;

  appendMessage({ role: "user", content: question });
  state.history.push({ role: "user", content: question });
  el.chatInput.value = "";
  autoResizeTextarea();
  setLoading(true);

  try {
    const response = await fetch(API.chat, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        mode: state.mode,
        history: state.history.slice(0, -1).slice(-10), // exclude the just-added message, cap context
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || data.detail || "The request failed.");
    }

    appendMessage({
      role: "assistant",
      content: data.answer,
      modeUsed: data.mode_used,
      sources: data.sources,
      webSources: data.web_sources,
    });
    state.history.push({ role: "assistant", content: data.answer });
  } catch (err) {
    appendMessage({
      role: "assistant",
      content: `Something went wrong: ${err.message}`,
      isError: true,
    });
  } finally {
    setLoading(false);
  }
});

el.clearChatBtn.addEventListener("click", () => {
  state.history = [];
  el.chatWindow.innerHTML = `
    <div class="welcome-message">
      <h2>How can I help you today?</h2>
      <p>Ask me anything, or upload a PDF and ask questions about it.</p>
    </div>`;
});

// ---------------------------- Document upload ----------------------------

el.fileInput.addEventListener("change", async () => {
  const file = el.fileInput.files[0];
  if (!file) return;

  el.uploadStatus.textContent = `Uploading "${file.name}"...`;
  el.uploadStatus.className = "upload-status";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(API.upload, { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || data.detail || "Upload failed.");
    }

    el.uploadStatus.textContent = `✓ ${data.document.document_name} indexed (${data.document.num_chunks} chunks)`;
    el.uploadStatus.className = "upload-status success";
    await loadDocuments();
  } catch (err) {
    el.uploadStatus.textContent = `✗ ${err.message}`;
    el.uploadStatus.className = "upload-status error";
  } finally {
    el.fileInput.value = "";
  }
});

async function loadDocuments() {
  try {
    const response = await fetch(API.documents);
    const documents = await response.json();
    state.documents = documents;
    renderDocumentList();
  } catch (err) {
    console.error("Failed to load documents", err);
  }
}

function renderDocumentList() {
  if (state.documents.length === 0) {
    el.documentList.innerHTML = `<p class="empty-hint">No documents uploaded yet.</p>`;
    return;
  }

  el.documentList.innerHTML = "";
  state.documents.forEach((doc) => {
    const item = document.createElement("div");
    item.className = "document-item";
    const sizeKb = Math.round(doc.size_bytes / 1024);
    item.innerHTML = `
      <div class="doc-info">
        <div class="doc-name" title="${escapeHtml(doc.document_name)}">${escapeHtml(doc.document_name)}</div>
        <div class="doc-meta">${doc.num_chunks} chunks · ${sizeKb} KB</div>
      </div>
      <button class="delete-btn" data-id="${doc.document_id}" title="Delete document">🗑️</button>
    `;
    el.documentList.appendChild(item);
  });

  el.documentList.querySelectorAll(".delete-btn").forEach((btn) => {
    btn.addEventListener("click", () => deleteDocument(btn.dataset.id));
  });
}

async function deleteDocument(documentId) {
  try {
    const response = await fetch(`${API.documents}/${documentId}`, { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || data.detail || "Failed to delete document.");
    }
    await loadDocuments();
  } catch (err) {
    alert(`Could not delete document: ${err.message}`);
  }
}

// ---------------------------- Image generation ----------------------------

el.imageForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const prompt = el.imagePrompt.value.trim();
  if (!prompt) return;

  clearWelcomeMessage();
  appendMessage({ role: "user", content: `🎨 Generate image: ${prompt}` });
  el.imagePrompt.value = "";
  el.imageGenerateBtn.disabled = true;
  el.imageGenerateBtn.textContent = "Generating...";
  setLoading(true);

  try {
    const response = await fetch(API.imageGenerate, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || data.detail || "Image generation failed.");
    }

    clearWelcomeMessage();
    const wrapper = document.createElement("div");
    wrapper.className = "message assistant";
    wrapper.innerHTML = `
      <div class="mode-tag">🎨 Image Generation</div>
      <div class="bubble generated-image-wrapper">
        <img src="${data.image_url}" alt="${escapeHtml(prompt)}" loading="lazy" />
        <div class="image-caption">${escapeHtml(prompt)}</div>
      </div>
    `;
    el.chatWindow.appendChild(wrapper);
    el.chatWindow.scrollTop = el.chatWindow.scrollHeight;
  } catch (err) {
    appendMessage({ role: "assistant", content: `Image generation failed: ${err.message}`, isError: true });
  } finally {
    setLoading(false);
    el.imageGenerateBtn.disabled = false;
    el.imageGenerateBtn.textContent = "🎨 Generate Image";
  }
});

// ---------------------------- Health check ----------------------------

async function checkHealth() {
  try {
    const response = await fetch(API.health);
    const data = await response.json();
    el.healthIndicator.classList.remove("ok", "degraded", "error");
    if (data.status === "ok") {
      el.healthIndicator.classList.add("ok");
      el.healthIndicator.title = "All systems operational";
    } else {
      el.healthIndicator.classList.add("degraded");
      el.healthIndicator.title = (data.warnings || []).join(" | ") || "Degraded";
    }
  } catch (err) {
    el.healthIndicator.classList.add("error");
    el.healthIndicator.title = "Backend unreachable";
  }
}

// ---------------------------- Init ----------------------------

loadDocuments();
checkHealth();
setInterval(checkHealth, 30000);
