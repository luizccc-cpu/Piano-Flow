# KeyFlow AI Audio-to-MIDI Backend (`backend/`)

This Python backend powers **Studio AI Dedicated Piano Stem Separation (`HT-Demucs 6S`)** and **Neural Polyphonic Piano Transcription (`ByteDance Piano Transcription` / `Basic Pitch` / `Librosa CQT`)** with **Post-HPSS Harmonic Cleaning & MIDI Bleed Filtering** for the KeyFlow Virtual Piano iPad & Web application.

> **Offline Fallback:** Even when no Python backend is running, the KeyFlow frontend includes an **In-Browser Web Audio API Polyphonic Piano Transcriber** (`js/audio-transcriber.js`), allowing `.mp3`, `.wav`, `.m4a`, and `.ogg` files to be transcribed 100% offline directly on iPad via GitHub Pages.

---

## Architecture & Audio Pipeline

1. **6-Stem Dedicated Piano Isolation (`HT-Demucs 6S`):**
   - Uses `htdemucs_6s --two-stems piano` to separate a dedicated piano track from guitars, bass, drums, vocals, and synths (falling back to 4-stem `htdemucs` and Librosa HPSS if unavailable).
2. **Post-Demucs Harmonic & Bandpass Cleaning (`clean_isolated_piano_audio`):**
   - Strips residual percussive bleed via Librosa HPSS (`margin=(1.0, 2.2)`).
   - Applies a piano fundamental bandpass filter (`27.5 Hz` A0 to `4400 Hz` C8) via `scipy.signal.butter` + `sosfiltfilt` to remove sub-bass rumble and high-hat hiss.
   - Peak normalizes audio prior to neural transcription.
3. **High-Resolution Transcription & Bleed Filtering (`filter_midi_bleed_notes`):**
   - Transcribes with ByteDance CRNN high-resolution onsets & frames.
   - Post-filters out-of-range notes (outside MIDI 21..108), micro-glitches (`< 55ms`), and background bleed under the sensitivity-adaptive velocity threshold (default 55% sensitivity -> velocity < 28 removed).
   - Caps simultaneous note onsets within any 30ms window to 8 notes (preserving highest-velocity notes).

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
   - The server will report active engines (`HT-Demucs 6S`, `ByteDance Piano AI`) on ZeroGPU.

---

## API Endpoints

- **`GET /health`** — Returns server status, version, and available AI engines (`HT-Demucs-6S`, `ByteDance-Piano`, `Basic-Pitch`, `Librosa-CQT`).
- **`POST /api/transcribe`** — Accepts `multipart/form-data`:
  - `file`: Audio file (`.mp3`, `.wav`, `.m4a`, `.aac`, `.ogg`, `.flac`, `.webm`, max 50 MB).
  - `isolate_piano`: `"true"` (default) to run HT-Demucs 6S dedicated piano isolation or `"false"`.
  - `sensitivity`: integer/float `15..85` (default `55`), normalized to `0.15..0.90`.
  - Returns `audio/midi` (Standard MIDI binary) or JSON note stream (`{"title": "...", "bpm": 110, "notes": [...]}` or `{"midi_base64": "...", "title": "..."}`).
