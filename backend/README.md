# KeyFlow AI Audio-to-MIDI Backend (`backend/`)

This Python backend powers **Studio AI Stem Separation (`HT-Demucs`)** and **Neural Polyphonic Piano Transcription (`ByteDance Piano Transcription` / `Basic Pitch` / `Librosa CQT`)** for the KeyFlow Virtual Piano iPad & Web application.

> **Offline Fallback:** Even when no Python backend is running, the KeyFlow frontend includes an **In-Browser Web Audio API Polyphonic Piano Transcriber** (`js/audio-transcriber.js`), allowing `.mp3`, `.wav`, `.m4a`, and `.ogg` files to be transcribed 100% offline directly on iPad via GitHub Pages.

---

## Deployment Options

You can run the backend locally on your computer or deploy it as a free **Hugging Face ZeroGPU Gradio Space**.

### Option A: Local FastAPI Server (`server.py`)

1. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
   *(Optional)* For full neural stem isolation and ByteDance CRNN piano transcription, install:
   ```bash
   pip install demucs basic-pitch piano-transcription-inference
   ```

2. **Start the server:**
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

3. **Connect from KeyFlow:**
   - From the KeyFlow **Home Screen**, click **`+ AI Audio Song`**.
   - If running on the same PC: keep the server URL as `http://localhost:8000`.
   - If practicing on an **iPad over Wi-Fi**: enter your PC's local LAN IP (e.g. `http://192.168.1.50:8000`) and tap **Check**.
   - Select an audio song and tap **Convert Audio to Piano Notes**.

---

### Option B: Free Hugging Face ZeroGPU Space (`app.py`)

Hugging Face provides free ZeroGPU hardware (NVIDIA RTX Pro 6000 Blackwell) using the Gradio SDK:

1. **Create a Gradio Space on Hugging Face:**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - Space SDK: **Gradio**.
   - Hardware: **ZeroGPU (Free)**.

2. **Upload the contents of `backend/`:**
   - Upload `app.py`, `packages.txt`, and `requirements.txt` to the root of your Hugging Face Space repository.
   - Hugging Face automatically installs system Debian packages (`ffmpeg`, `libsndfile1`) and starts `backend/app.py`.

3. **Connect from KeyFlow:**
   - Copy your public Space URL (e.g. `https://<username>-<space-name>.hf.space`).
   - Open KeyFlow, tap **`+ AI Audio Song`**, paste the `.hf.space` URL into the **AI Server URL** field, and tap **Check**.
   - The server will report active engines (`HT-Demucs`, `ByteDance Piano AI`) on ZeroGPU.

---

## API Endpoints

- **`GET /health`** — Returns server status, version, and available AI engines (`HT-Demucs`, `ByteDance-Piano`, `Basic-Pitch`, `Librosa-CQT`).
- **`POST /api/transcribe`** — Accepts `multipart/form-data`:
  - `file`: Audio file (`.mp3`, `.wav`, `.m4a`, `.aac`, `.ogg`, `.flac`, `.webm`, max 50 MB).
  - `isolate_piano`: `"true"` (default) to run HT-Demucs stem isolation or `"false"`.
  - `sensitivity`: integer/float `15..85` (default `55`), normalized to `0.15..0.90`.
  - Returns `audio/midi` (Standard MIDI binary) or JSON note stream (`{"title": "...", "bpm": 110, "notes": [...]}` or `{"midi_base64": "...", "title": "..."}`).
