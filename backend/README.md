# KeyFlow AI Audio-to-MIDI Backend (`backend/`)

This optional Python FastAPI server powers **Studio AI Stem Separation (`HT-Demucs`)** and **Neural Polyphonic Piano Transcription (`ByteDance Piano Transcription` / `Basic Pitch` / `Librosa CQT`)** for the KeyFlow Virtual Piano app.

> **Note:** Even when this Python server is not running, the KeyFlow web app automatically falls back to its built-in **In-Browser Web Audio API Polyphonic Piano Transcriber** (`js/audio-transcriber.js`), so `.mp3`, `.wav`, `.m4a`, and `.ogg` transcription works 100% offline on iPad via GitHub Pages.

---

## Quick Start (3 Steps)

### 1. Install Core Dependencies
```bash
cd backend
pip install -r requirements.txt
```

*(Optional)* For full neural stem separation and studio piano transcription, also install:
```bash
pip install demucs basic-pitch piano-transcription-inference
```

### 2. Start the FastAPI Server
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 3. Connect from KeyFlow (PC or iPad on Wi-Fi)
1. Open KeyFlow and click **AI Audio** in the top bar.
2. If running on the same computer, leave the server URL as `http://localhost:8000`.
3. If using your **iPad on the same Wi-Fi network**, enter your computer's local LAN IP (for example `http://192.168.1.25:8000`) and tap **Check**.
4. Select any `.MP3`, `.WAV`, `.M4A`, or `.OGG` song and tap **Convert Audio to Piano Notes**.

---

## API Endpoints

- **`GET /health`** — Returns active AI engines (`HT-Demucs`, `ByteDance-Piano`, `Basic-Pitch`, `Librosa-CQT`).
- **`POST /api/transcribe`** — Accepts `multipart/form-data` (`file`, `isolate_piano`, `sensitivity`) and returns a Standard MIDI file (`audio/midi`) or JSON note array, which the KeyFlow frontend parses and splits into playable Left/Right hand parts via `assignHandsSmart`.
