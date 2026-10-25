# Voice Agent

A real-time voice assistant deployed on Google Cloud Run. Hold a button, speak, and get a spoken response back — fully in the browser, no installs required.

**Live demo:** https://voiceagent-724061677236.us-central1.run.app

> **Note:** First load may take ~30 seconds (Cloud Run cold start while the Whisper model loads into memory). Subsequent requests are fast.

---

## How it works

1. **Browser** records audio via the MediaRecorder API (WebM/Opus) and POSTs it to `/talk`
2. **ffmpeg** converts the WebM to 16kHz mono WAV
3. **Faster-Whisper** (`small`, int8, CPU) transcribes the audio
4. **Gemini 2.5 Flash** (via Google ADK) generates a response
5. **Google Cloud Text-to-Speech** (Neural2-F, en-US) synthesizes the reply as MP3
6. Everything is returned in one JSON response — transcript, detected language, text response, and base64-encoded audio — which the frontend decodes and plays back

---

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| STT | faster-whisper (small model, CPU int8) |
| LLM | Gemini 2.5 Flash via Google ADK |
| TTS | Google Cloud Text-to-Speech (Neural2-F) |
| Audio processing | ffmpeg |
| Deployment | Docker → Google Cloud Run |
| Package manager | uv |

---

## Architecture notes

- **Thread safety:** Whisper is not thread-safe. A `Semaphore(1)` serialises all transcription requests against the single shared model instance.
- **Async safety:** All blocking work (ffmpeg, Whisper, TTS) runs in a `ThreadPoolExecutor` via `run_in_executor` — the async event loop is never blocked.
- **TTS async conflict:** The Google TTS gRPC async transport creates its own internal event loop at import time, which conflicts with uvicorn's loop. The sync `TextToSpeechClient` + thread pool avoids this entirely.
- **No response headers for large payloads:** Transcript + audio were originally returned as custom HTTP headers (`X-Transcript`, `X-Assistant-Response`). Headers have a strict per-header size cap (~8KB) that Google's Cloud Run frontend silently enforces, dropping long transcripts entirely. Everything now returns as a single JSON body, governed only by Cloud Run's 32MB response limit.
- **Audio size limit:** Supports audio files up to ~22 minutes at 192kbps, constrained by Cloud Run's 32MB request body ceiling.

---

## Running locally

```bash
# Clone and install
git clone https://github.com/<your-username>/voiceagent
cd voiceagent
pip install uv
uv pip install --system .

# Set up GCP credentials (needed for TTS and Gemini ADK)
gcloud auth application-default login

# Run
uvicorn server:app --host 0.0.0.0 --port 8080
```

Open http://localhost:8080 and hold the button to talk.

---

## Deploying to Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/<PROJECT_ID>/voiceagent

# Deploy
gcloud run deploy voiceagent \
  --image gcr.io/<PROJECT_ID>/voiceagent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

Requires a GCP project with Cloud Run, Cloud Build, and Text-to-Speech APIs enabled, and a service account with Gemini / Vertex AI access.

---

## Project structure

```
voiceagent/
├── server.py        # FastAPI app — STT → LLM → TTS pipeline
├── index.html       # Frontend — hold-to-talk UI
├── Dockerfile       # Python 3.11-slim + ffmpeg + uv
├── pyproject.toml   # Dependencies
└── .dockerignore
```
