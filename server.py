import uuid
import subprocess
import os
import tempfile
import asyncio
import base64
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from faster_whisper import WhisperModel

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.cloud import texttospeech


# -------------------
# THREAD POOL
# Used to offload blocking CPU/IO work off the async event loop.
# max_workers=2 is intentional for Cloud Run: keep it <= your container's
# vCPU count so you don't over-subscribe. Raise it if you bump CPU alloc.
# -------------------
_thread_pool = ThreadPoolExecutor(max_workers=2)


# -------------------
# WHISPER — loaded once, guarded by a semaphore.
# The semaphore ensures only one transcription runs at a time so the shared
# WhisperModel is never accessed concurrently (it is not thread-safe).
# Requests queue here instead of crashing or corrupting results.
# -------------------
print("Loading Faster-Whisper...")

whisper_model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

# Allow 1 concurrent transcription. Raise to N only if you switch to a
# model pool with N instances.
_whisper_sem = asyncio.Semaphore(1)


# -------------------
# GEMINI AGENT
# -------------------
agent = Agent(
    name="voice_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant. Keep responses short."
)

APP_NAME = "voice_agent_app"
USER_ID = "default_user"

session_service = InMemorySessionService()
runner = Runner(
    agent=agent,
    app_name=APP_NAME,
    session_service=session_service
)


# -------------------
# TTS — sync client, called via run_in_executor.
#
# WHY NOT AsyncClient: google-cloud-texttospeech's gRPC async transport
# creates its own internal event loop at import time. When uvicorn's loop
# then tries to await a Future from that loop, Python raises:
#   "Task got Future attached to a different loop"
# The safe pattern on Cloud Run is the sync client + thread pool.
# -------------------
tts = texttospeech.TextToSpeechClient()


def _synthesize_speech(response_text: str) -> bytes:
    """Synchronous TTS call — must be run in the thread pool."""
    synthesis = texttospeech.SynthesisInput(text=response_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-F",
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
    )
    response = tts.synthesize_speech(
        input=synthesis,
        voice=voice,
        audio_config=audio_config,
    )
    return response.audio_content


# -------------------
# LIFESPAN — clean shutdown of the thread pool.
# -------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _thread_pool.shutdown(wait=False)


# -------------------
# APP
# -------------------
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------
# HELPERS
# -------------------
def _convert_to_wav(input_path: str, output_path: str) -> None:
    """Synchronous ffmpeg call — must be run in the thread pool, not on the event loop."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            output_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _transcribe_audio(path: str) -> tuple[str, str]:
    """Synchronous Whisper inference — must be run in the thread pool."""
    segments, info = whisper_model.transcribe(
        path,
        beam_size=1,
        vad_filter=True,
    )
    text = "".join(segment.text for segment in segments)
    return text, info.language


async def convert_to_wav(input_path: str, output_path: str) -> None:
    """Async wrapper: runs ffmpeg in the thread pool so the event loop stays free."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_thread_pool, _convert_to_wav, input_path, output_path)


async def transcribe_audio(path: str) -> tuple[str, str]:
    """Async wrapper: acquires the semaphore then runs Whisper in the thread pool."""
    loop = asyncio.get_running_loop()
    async with _whisper_sem:                          # serialise access to the model
        return await loop.run_in_executor(_thread_pool, _transcribe_audio, path)


# -------------------
# MAIN ENDPOINT
# -------------------
@app.post("/talk")
async def talk(file: UploadFile = File(...)):

    session_id = str(uuid.uuid4())

    # Cloud Run safe: tempfile.TemporaryDirectory() lands in /tmp (in-memory
    # tmpfs on Cloud Run, so it's fast and cleaned up automatically).
    with tempfile.TemporaryDirectory() as tmp:

        webm_path = os.path.join(tmp, "audio.webm")
        wav_path  = os.path.join(tmp, "audio.wav")

        # Save upload
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty audio file received.")
        with open(webm_path, "wb") as f:
            f.write(raw)

        # -------------------
        # CONVERT (non-blocking)
        # -------------------
        try:
            await convert_to_wav(webm_path, wav_path)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=422, detail=f"Audio conversion failed: {e}")

        # -------------------
        # STT — non-blocking, serialised through semaphore
        # -------------------
        text, detected_lang = await transcribe_audio(wav_path)

        print("Detected language:", detected_lang)
        print("User:", text)

        if not text.strip():
            raise HTTPException(status_code=422, detail="No speech detected in audio.")

        # -------------------
        # SESSION
        # -------------------
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )

        content = types.Content(
            role="user",
            parts=[types.Part(text=text)],
        )

        # -------------------
        # LLM (GEMINI) — already async
        # -------------------
        response_text = ""

        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=content,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                response_text = event.content.parts[0].text

        print("Assistant:", response_text)

        if not response_text:
            raise HTTPException(status_code=500, detail="LLM returned an empty response.")

        # -------------------
        # TTS — sync client in thread pool (avoids gRPC loop conflict)
        # -------------------
        loop = asyncio.get_running_loop()
        audio_content = await loop.run_in_executor(
            _thread_pool, _synthesize_speech, response_text
        )

        # -------------------
        # RETURN JSON (audio as base64)
        #
        # WHY NOT HEADERS: we previously returned the MP3 in the response
        # body and put the transcript/response text in custom headers
        # (X-Transcript, X-Assistant-Response). That breaks for longer
        # audio: HTTP headers have a strict per-header size cap (commonly
        # ~8KB, sometimes less depending on the proxy), and Google's
        # frontend in front of Cloud Run will silently drop or truncate
        # oversized headers. A 10-minute transcript easily exceeds that,
        # which is why X-Transcript disappeared entirely on longer files.
        #
        # Putting everything in the JSON body removes that limit — JSON
        # body size is governed by Cloud Run's request/response size
        # limit (32MB), not the much smaller per-header cap.
        # -------------------
        audio_base64 = base64.b64encode(audio_content).decode("utf-8")

        return JSONResponse({
            "transcript": text,
            "detected_language": detected_lang,
            "response": response_text,
            "audio_base64": audio_base64,   # decode + play as audio/mpeg on the client
        })


# -------------------
# HEALTH CHECK — Cloud Run uses this to verify the container is up.
# -------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------
# FRONTEND
# -------------------
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()