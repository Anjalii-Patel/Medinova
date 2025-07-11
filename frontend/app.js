// frontend/app.js
const chatBox = document.getElementById("chat-box");
const uploadedDocsContainer = document.getElementById("uploaded-documents");

// Load or create session ID
let sessionId = localStorage.getItem("session_id") || `sess_${Date.now()}`;
localStorage.setItem("session_id", sessionId);
document.getElementById("session_id").value = sessionId;

function appendMessage(sender, text) {
  const div = document.createElement("div");
  div.className = "message " + sender;
  div.innerText = `${sender === "user" ? "You" : "Bot"}: ${text}`;
  chatBox.appendChild(div);
  div.scrollIntoView({ behavior: "smooth" });
}

async function uploadDoc() {
  const file = document.getElementById("docFile").files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);
  formData.append("session_id", sessionId);

  document.getElementById("uploadStatus").innerText = "Uploading...";

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });

    if (!res.ok) {
      const err = await res.json();
      document.getElementById("uploadStatus").innerText = (err.detail || "Upload failed.");
      return;
    }

    const data = await res.json();
    document.getElementById("uploadStatus").innerText = data.status;
    await loadDocuments();
  } catch (err) {
    document.getElementById("uploadStatus").innerText = "Upload failed.";
    console.error(err);
  }
}

async function loadDocuments() {
  try {
    const res = await fetch(`/documents/${sessionId}`);
    const docs = await res.json();
    uploadedDocsContainer.innerHTML = docs.map(doc => `
      <div class="doc-entry">
        <span>${doc}</span>
        <i class="fas fa-trash" onclick="deleteDocument('${doc}')"></i>
      </div>
    `).join('');
  } catch {
    uploadedDocsContainer.innerHTML = '<p class="status">Failed to load documents.</p>';
  }
}

async function deleteDocument(docName) {
  const formData = new FormData();
  formData.append("session_id", sessionId);
  formData.append("filename", docName); // must match backend param
  const res = await fetch(`/delete_doc`, {
    method: "POST",
    body: formData
  });
  if (res.ok) await loadDocuments();
}

let isBotProcessing = false;

async function ask(inputText = null) {
  if (isBotProcessing) return;
  isBotProcessing = true;

  const qInput = document.getElementById("question");
  const sendBtn = document.getElementById("sendButton");
  const sendIcon = document.getElementById("sendIcon");

  const question = inputText ?? qInput.value.trim();
  if (!question) { isBotProcessing = false; return; }

  sendBtn.disabled = true;
  sendIcon.className = "fas fa-spinner fa-spin";

  appendMessage("user", question);
  qInput.value = "";

  const typing = document.createElement("div");
  typing.className = "typing-indicator";
  typing.id = "typing-indicator";
  typing.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span>`;
  chatBox.appendChild(typing);
  typing.scrollIntoView({ behavior: "smooth" });

  const formData = new FormData();
  formData.append("question", question);
  formData.append("session_id", sessionId);

  try {
    const res = await fetch("/ask", { method: "POST", body: formData });
    const data = await res.json();

    document.getElementById("typing-indicator").remove();
    appendMessage("bot", data.response || "No response.");
  } catch {
    document.getElementById("typing-indicator").remove();
    appendMessage("bot", "Error connecting to backend.");
  } finally {
    sendBtn.disabled = false;
    qInput.disabled = false;
    sendIcon.className = "fas fa-paper-plane";
    qInput.focus();
    isBotProcessing = false;
  }
}

// ===================== MIC & ASR =====================
let socket, mediaRecorder, micOn = false, lastTranscript = "", debounceTimeout;

function toggleMic() {
  const micBtn = document.getElementById("micToggle");
  const micIcon = document.getElementById("micIcon");

  if (micOn) {
    micIcon.className = "fas fa-microphone-slash";
    mediaRecorder.stop();
    socket.close();
    micOn = false;
    document.getElementById("micStatus").innerText = "Mic is OFF";
    const finalText = document.getElementById("question").value.trim();
    if (finalText && !isBotProcessing) ask(finalText);
    return;
  }

  micBtn.disabled = true;
  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    socket = new WebSocket("ws://localhost:8000/ws/asr");
    socket.onmessage = e => {
      const transcript = e.data.trim();
      if (!transcript || transcript === lastTranscript) return;
      lastTranscript = transcript;
      clearTimeout(debounceTimeout);
      debounceTimeout = setTimeout(() => {
        document.getElementById("question").value = transcript;
      }, 400);
    };

    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0 && socket.readyState === WebSocket.OPEN) {
        e.data.arrayBuffer().then(buff => socket.send(buff));
      }
    };
    mediaRecorder.start(250);
    micOn = true;
    micBtn.disabled = false;
    micIcon.className = "fas fa-microphone";
    document.getElementById("micStatus").innerText = "Mic is ON and streaming...";
  }).catch(() => {
    micBtn.disabled = false;
    document.getElementById("micStatus").innerText = "Mic error";
  });
}

// ===================== CHAT SESSIONS =====================
// Custom modal for chat deletion
function showDeleteChatModal(onConfirm) {
  let modal = document.getElementById("deleteChatModal");
  if (!modal) {
    modal = document.createElement("div");
    modal.id = "deleteChatModal";
    modal.innerHTML = `
      <div class="modal-overlay"></div>
      <div class="modal-content">
        <h3>Delete Chat</h3>
        <p>Are you sure you want to delete this chat? This action cannot be undone.</p>
        <div class="modal-actions">
          <button id="modalCancelBtn" class="modal-btn cancel">Cancel</button>
          <button id="modalDeleteBtn" class="modal-btn delete">Delete</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
  }
  modal.style.display = "flex";
  document.getElementById("modalCancelBtn").onclick = () => {
    modal.style.display = "none";
  };
  document.getElementById("modalDeleteBtn").onclick = () => {
    modal.style.display = "none";
    onConfirm();
  };
}

async function loadChatHistory() {
  try {
    const res = await fetch("/chats");
    const chats = await res.json();
    const container = document.getElementById("chatHistory");
    container.innerHTML = "";
    chats.forEach(c => {
      const div = document.createElement("div");
      div.classList.add("chat-item");
      div.textContent = c.preview || c.session_id;
      div.onclick = () => loadChat(c.session_id);

      const del = document.createElement("span");
      del.className = "delete-chat";
      del.innerHTML = '<i class="fas fa-trash"></i>';
      del.onclick = (e) => {
        e.stopPropagation();
        deleteChat(c.session_id);
      };
      div.appendChild(del);
      container.appendChild(div);
    });
  } catch {
    console.error("Failed loading chats");
  }
}

async function deleteChat(id) {
  showDeleteChatModal(async () => {
    const formData = new FormData();
    formData.append("session_id", id);
    const res = await fetch(`/delete_session`, { method: "POST", body: formData });
    if (res.ok) {
      if (id === sessionId) startNewChat();
      await loadChatHistory();
    }
  });
}

async function loadChat(id) {
  sessionId = id;
  localStorage.setItem("session_id", id);
  document.getElementById("session_id").value = id;
  chatBox.innerHTML = "";

  try {
    const res = await fetch(`/chat/${sessionId}`);
    const d = await res.json();

    if (d.messages.length === 0) {
      appendMessage("bot", "New chat started. Ask your question!");
    } else {
      d.messages.forEach(m => appendMessage(m.role, m.text));
    }
    await loadDocuments();
  } catch {
    console.error("Load chat failed");
  }
}

function startNewChat() {
  sessionId = `sess_${Date.now()}`;
  localStorage.setItem("session_id", sessionId);
  document.getElementById("session_id").value = sessionId;
  chatBox.innerHTML = "";
  uploadedDocsContainer.innerHTML = "";
  appendMessage("bot", "New chat started. Ask your question!");
  loadChatHistory();
  loadDocuments(); // <-- Ensure document list is refreshed
}

window.onload = () => {
  loadChatHistory();
  loadChat(sessionId);  // <- restore last session
  document.getElementById("session_id").value = sessionId;

  document.getElementById("question").addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey && !isBotProcessing) {
      e.preventDefault();
      ask();
    }
  });
};
