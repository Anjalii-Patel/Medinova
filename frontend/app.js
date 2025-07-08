// frontend/app.js
const chatBox = document.getElementById("chat-box");

// Session handling
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
  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();
    document.getElementById("uploadStatus").innerText = data.status;
  } catch {
    document.getElementById("uploadStatus").innerText = "Upload failed.";
  }
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

async function loadChatHistory() {
  try {
    const res = await fetch("/chats");
    const chats = await res.json();
    const container = document.getElementById("chatHistory");
    container.innerHTML = "";
    chats.forEach(c => {
      const div = document.createElement("div");
      div.textContent = c.preview || c.session_id;
      div.onclick = () => loadChat(c.session_id);
      container.appendChild(div);
    });
  } catch {
    console.error("Failed loading chats");
  }
}

async function loadChat(id) {
  sessionId = id;
  localStorage.setItem("session_id", id);
  document.getElementById("session_id").value = id;
  chatBox.innerHTML = "";
  try {
    const res = await fetch(`/chat/${sessionId}`);
    const d = await res.json();
    d.messages.forEach(m => appendMessage(m.role, m.text));
  } catch {
    console.error("Load chat failed");
  }
}

function startNewChat() {
  sessionId = `sess_${Date.now()}`;
  localStorage.setItem("session_id", sessionId);
  document.getElementById("session_id").value = sessionId;
  chatBox.innerHTML = "";
  appendMessage("bot", "New chat started. Ask your question!");
  loadChatHistory();
}

window.onload = () => {
  loadChatHistory();
  document.getElementById("session_id").value = sessionId;
  document.getElementById("question").addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey && !isBotProcessing) {
      e.preventDefault();
      ask();
    }
  });
};
