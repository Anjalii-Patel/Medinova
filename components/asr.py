# components/asr.py
import os
import uuid
import asyncio
import numpy as np
import soundfile as sf
from io import BytesIO
from faster_whisper import WhisperModel
from starlette.websockets import WebSocket

# Init model
model = WhisperModel("medium", device="cpu", compute_type="int8")

# Directory to save audio
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Transcribe audio file (for /transcribe endpoint)
def transcribe_audio(file_path: str) -> str:
    segments, _ = model.transcribe(file_path, language="en", task="translate")
    return " ".join([s.text.strip() for s in segments])

# Async transcription for live WebSocket stream
async def stream_asr(websocket: WebSocket):
    session_id = str(uuid.uuid4())
    audio_chunks = []
    try:
        while True:
            data = await websocket.receive_bytes()
            
            # Convert PCM to float32 waveform
            if len(data) % 2 != 0:
                data += b'\x00'
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_chunks.append(audio_np)

            # Concatenate and transcribe
            if sum(len(chunk) for chunk in audio_chunks) >= 16000 * 3:  # 3 sec buffer
                full_audio = np.concatenate(audio_chunks[-5:])
                audio_chunks = audio_chunks[-5:]

                segments, _ = model.transcribe(full_audio, language="en", task="translate")
                text = " ".join(s.text.strip() for s in segments)
                await websocket.send_text(text)
    except Exception as e:
        print(f"WebSocket disconnected: {e}")
    finally:
        # Save full audio to file
        if audio_chunks:
            final_audio = np.concatenate(audio_chunks)
            sf.write(f"{AUDIO_DIR}/{session_id}.wav", final_audio, samplerate=16000)
