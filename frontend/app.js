// frontend/app.js
const chatBox = document.getElementById("chat-box");
const transcriptDisplay = document.getElementById("partialTranscript");

function appendMessage(sender, text) {
  const div = document.createElement("div");
  div.className = "message " + sender;
  div.innerText = `${sender === "user" ? "You" : "Bot"}: ${text}`;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
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
  } catch (err) {
    document.getElementById("uploadStatus").innerText = "Upload failed.";
    console.error("[uploadDoc] Error:", err);
  }
}

async function ask(inputText = null) {
  const questionInput = document.getElementById("question");
  const session = document.getElementById("session_id").value;
  const question = inputText || questionInput.value.trim();
  if (!question) return;

  appendMessage("user", question);
  questionInput.value = "";

  const typingDiv = document.createElement("div");
  typingDiv.className = "message bot";
  typingDiv.id = "typing-indicator";
  typingDiv.innerText = "Bot is typing...";
  chatBox.appendChild(typingDiv);
  chatBox.scrollTop = chatBox.scrollHeight;

  const formData = new FormData();
  formData.append("question", question);
  formData.append("session_id", session);

  try {
    const res = await fetch("/ask", { method: "POST", body: formData });
    const data = await res.json();

    const indicator = document.getElementById("typing-indicator");
    if (indicator) indicator.remove();

    appendMessage("bot", data.response || "No response.");
  } catch (err) {
    const indicator = document.getElementById("typing-indicator");
    if (indicator) indicator.remove();

    appendMessage("bot", "Error connecting to backend.");
    console.error("[ask] Error:", err);
  }
}

let socket, mediaRecorder, micOn = false, lastTranscript = "", debounceTimeout;

async function toggleMic() {
  const micIcon = document.getElementById("micIcon");
  if (!micOn) {
    micIcon.className = "fas fa-microphone";
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    socket = new WebSocket("ws://localhost:8000/ws/asr");

    socket.onopen = () => console.log("WebSocket connected");

    socket.onmessage = (event) => {
      const transcript = event.data.trim();
      if (!transcript || transcript === lastTranscript) return;
      lastTranscript = transcript;

      clearTimeout(debounceTimeout);
      debounceTimeout = setTimeout(() => {
        document.getElementById("question").value = transcript;
        transcriptDisplay.innerText = transcript;
      }, 400);
    };

    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    mediaRecorder.ondataavailable = async (e) => {
      if (e.data.size > 0 && socket.readyState === WebSocket.OPEN) {
        const arrayBuffer = await e.data.arrayBuffer();
        socket.send(arrayBuffer);
      }
    };

    mediaRecorder.start(250);
    micOn = true;
    document.getElementById("micStatus").innerText = "Mic is ON and streaming...";
    transcriptDisplay.innerText = "...";

  } else {
    micIcon.className = "fas fa-microphone-slash";
    mediaRecorder.stop();
    socket.close();
    micOn = false;
    document.getElementById("micStatus").innerText = "Mic is OFF";

    const finalText = document.getElementById("question").value.trim();
    if (finalText) {
      transcriptDisplay.innerText = "Finalizing...";
      setTimeout(() => {
        ask(finalText);
        transcriptDisplay.innerText = "...";
        document.getElementById("question").value = "";
      }, 800);
    }
  }
}

async function loadChatHistory() {
  try {
    const res = await fetch("/chats");
    const chats = await res.json();
    const container = document.getElementById("chatHistory");
    container.innerHTML = "";
    chats.forEach(chat => {
      const div = document.createElement("div");
      div.textContent = chat.preview || chat.session_id;
      div.onclick = () => loadChat(chat.session_id);
      container.appendChild(div);
    });
  } catch (err) {
    console.error("[loadChatHistory] Error:", err);
  }
}

async function loadChat(sessionId) {
  document.getElementById("session_id").value = sessionId;
  try {
    const res = await fetch(`/chat/${sessionId}`);
    const data = await res.json();
    chatBox.innerHTML = "";
    data.messages.forEach(m => appendMessage(m.role, m.text));
  } catch (err) {
    console.error("[loadChat] Error:", err);
  }
}

function startNewChat() {
  const newSessionId = `sess_${Date.now()}`;
  document.getElementById("session_id").value = newSessionId;
  chatBox.innerHTML = "";
  appendMessage("bot", "New chat started. Ask your question!");
}

window.onload = loadChatHistory;
