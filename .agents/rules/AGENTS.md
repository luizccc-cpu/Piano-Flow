---
trigger: always_on
description: "Core architecture, UX constraints, and phased roadmap for the Falling Notes Virtual Piano app"
---

# Project Blueprint: Falling Notes Virtual Piano (iPad Edition)

## Core UX & Audio Rules
- **Target Device:** iPad Landscape orientation (supports Add to Home Screen fullscreen mode).
- **Constant 5-Second Horizon:** Notes must always spawn at the top of the canvas exactly 5.0 seconds before impact (`NOW` line), regardless of song BPM or playback speed multiplier.
- **Readable Falling Bars:** Every falling bar must render its note name and octave badge (e.g., `F3`, `C4`) inside the leading edge of the bar.
- **Auto-Scaled Keyboard:** Dynamically size the virtual keyboard to the exact octave range of the active song (minimum 2 octaves, maximum 88 keys), with C-octave labels on the keys.
- **Audio Engine:** Default to the "Acoustic Grand Piano" preset (with "Mellow Felt Grand" second on the list) and a live Felt Softness / Warmth low-pass & transient shaper slider (0%–100%).

## Development Phases
- **Phase 1 (Complete):** Core visualizer, 5s horizon, auto-scaled keyboard, Mellow Felt Grand sound engine, A-B looping, Hand Isolation (L/R), Wait Mode, and `.MID` file upload.
- **Phase 2 (Complete):** 
  - Smart A-B Loop Tempo Ramp (start loop at 50% speed and auto-increment by +5% after each clean pass).
  - Advanced Left/Right Hand voice-splitting algorithm using pitch clustering and human hand-span constraints (max 10th interval per hand) for single-track MIDI files.
  - Measure & beat gridlines synchronized to MIDI time signatures.
- **Phase 3 (Complete):** Hybrid AI Audio-to-MIDI Transcription Pipeline ("Bring Your Own Song") featuring a Python FastAPI / Hugging Face ZeroGPU Gradio backend (`backend/app.py`, `backend/server.py`) running `HT-Demucs` (stem separation) + `ByteDance Piano Transcription` / `Basic Pitch` / `Librosa CQT`, paired with an In-Browser Web Audio API Polyphonic Piano Transcriber (`js/audio-transcriber.js`), Persistent `IndexedDB` Custom Song Library, and Standard `.MID` Exporter.
- **Phase 4A (Complete):**
  - Full Home Screen Song Library view (`#homeView`) with search, source filter pills (`All`, `AI Audio`, `MIDI Uploads`, `Built-In Classics`), and song cards (`Play`, `Edit Notes`, `.MID` export, `Delete`/`Reset`).
  - Decluttered Playing / Practice top bar (`← Home`, song title, and `✎ Edit Notes` toggle).
  - Interactive Visual Note Editor (Edit Mode) directly on the 5-second falling-note runway (`#pianoCanvas`) with Select/Move/Resize, Add Note, Erase, Snap-to-Grid (`Free`, `1/4`, `1/8`, `1/16`), L/R hand flip, semitone/octave transpose, BPM & Bar 1 grid alignment, 30-step Undo/Redo, and automatic persistence to `IndexedDB` (`KeyFlowSongDB`).
- **Phase 4B (Next):**
  - Live Chord Symbol Detector (`C`, `G/B`, `Am7`, `Fmaj7` displayed in real time).
  - Audible Metronome & Count-In (`1-2-3-4` pre-roll before playback or loop start).
  - Live MIDI Accuracy & Wrong-Note Feedback + End-of-Pass Score Summary.
  - Editable Finger Number Badges (`1–5`) on falling notes.
  - Full Library `.keyflow.json` Backup & Restore.
