# components/asr.py
import os
import io
import uuid
import asyncio
import soundfile as sf
from faster_whisper import WhisperModel
from starlette.websockets import WebSocket

# Initialize Whisper model
model = WhisperModel("medium", device="cpu", compute_type="int8")

# Audio save path
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Store audio chunks per session
sessions = {}

def transcribe_audio(file_path: str) -> str:
    segments, _ = model.transcribe(file_path, language="en", task="translate")
    return " ".join([s.text.strip() for s in segments])

async def stream_asr(websocket: WebSocket):
    # await websocket.accept()
    session_id = str(uuid.uuid4())
    sessions[session_id] = []  # list of bytes

    try:
        while True:
            data = await websocket.receive_bytes()
            sessions[session_id].append(data)

            # Transcribe short audio chunk
            audio_buffer = b"".join(sessions[session_id][-5:])  # last few chunks
            try:
                audio_np, sr = sf.read(io.BytesIO(audio_buffer))
                segments, _ = model.transcribe(audio_np, language="en", task="translate")
                text = " ".join([s.text.strip() for s in segments])
                await websocket.send_text(text)
            except Exception as e:
                await websocket.send_text(f"(partial error: {e})")

    except Exception as e:
        print(f"WebSocket disconnected: {e}")
    finally:
        # Save full audio
        full_bytes = b"".join(sessions[session_id])
        audio_path = os.path.join(AUDIO_DIR, f"{session_id}.mp3")
        with open(audio_path, "wb") as f:
            f.write(full_bytes)
        del sessions[session_id]
