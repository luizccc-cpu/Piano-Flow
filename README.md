# 🎹 KeyFlow (Piano-Flow) — 5-Second Horizon Virtual Piano

**KeyFlow** is an iPad-first, zero-dependency web application designed to make learning piano songs faster, clearer, and more intuitive than reading traditional sheet music or pausing YouTube falling-note videos.

Built for a landscape tablet resting on your piano's music stand, KeyFlow pairs a **constant 5-second lookahead runway**, **dynamic octave auto-scaling**, and **explicit note badges inside every falling bar** with a **multi-sampled Acoustic Grand Piano audio engine**, **AI Audio-to-MIDI song transcription**, and an **Interactive Visual Note Editor**.

---

## ✨ Key Features

### 🏠 Home Screen Song Library & Clean Practice Header
* **Full Song Library (`#homeView`):** Browse Built-In Classics, uploaded `.MID` songs, and AI-transcribed audio tracks with instant search and category filter pills (`All Songs`, `AI Audio`, `MIDI Uploads`, `Built-In Classics`).
* **One-Tap Card Actions:** Launch any song directly into **Play** mode, open it in **Edit Notes** mode, export a 2-track Standard `.MID` file, or delete/reset custom edits.
* **Decluttered Practice Header:** Keeps the playing view distraction-free with only `← Home`, the active song title, and `✎ Edit Notes`.

### ✏️ Interactive Visual Note Editor (Edit Mode)
* **Direct Canvas Editing on the 5s Runway:** Tap **✎ Edit Notes** at any moment to pause and edit notes right on the falling-note runway:
  * **Select, Move & Resize (`↖ Select / Resize`):** Tap any note to audition it, drag horizontally/vertically to adjust pitch and timing, or drag the bright **top resize handle** to lengthen or shorten note duration.
  * **Draw Notes (`＋ Add Note`):** Tap and drag upward on any key lane (or double-tap empty runway space) to draw new notes with live audio preview.
  * **Erase & Inspect (`✕ Erase` / Inspector Bar):** Flip any note between Left Hand (**Blue**) and Right Hand (**Green**), transpose by `±1` semitone or `±1` octave, align song BPM & `Bar 1` grid offset, and step through **30-level Undo / Redo** (`Ctrl+Z` / `Ctrl+Y`) with automatic saving to `IndexedDB`.

### 🎯 Visual Clarity Built for the Piano Bench
* **Constant 5-Second Lookahead Horizon:** Notes always appear at the top of the screen exactly **5.0 seconds** before reaching the `NOW` strike line (adjustable between `3.0s`, `5.0s`, and `7.0s`), regardless of the song's BPM or playback speed.
* **Note Names Inside Every Falling Bar:** Every descending bar displays a high-contrast badge (`F3`, `C4`, `A4`, `C5`) pinned to its leading edge so you can identify chords and intervals instantly without guessing.
* **Dynamic Auto-Scaled Keyboard:** Automatically scans the active song and scales the virtual piano to show only the octaves used in that piece (with a 2-octave minimum and 88-key maximum), keeping keys wide and readable. Switch anytime between **Auto-Scale Range**, **Min 2 Octaves**, and **Full 88 Keys**.
* **Scrolling Measure & Beat Gridlines:** Displays horizontal bar lines (`BAR 1`, `BAR 2`, ...) and beat subdivisions synced to the song's time signature alongside static `1s–5s` horizon markers.

### 🔊 Acoustic Grand & Mellow Felt Audio Engine
* **Acoustic Grand Piano (Default):** Crisp, balanced concert grand tone powered by background-loaded **Salamander Grand Piano** multisamples and a zero-latency Web Audio physical modeling engine (`24%` default warmth).
* **Curated Sound Palette:** Switch seamlessly in the left HUD between:
  * **Acoustic Grand Piano** *(Default)* — Balanced, resonant studio grand for classical, pop, and everyday practice.
  * **Mellow Felt Grand** — Intimate, dampened felt hammer attack (`78%` warmth) ideal for soft ballads and late-night sessions.
  * **Warm Vintage Upright** — Cozy, close-mic wooden soundboard resonance (`58%` warmth).
  * **Warm Electric Piano** — Rhodes-inspired tine bell and warm fundamental tone (`48%` warmth).
* **Live Felt Softness / Warmth Slider (`0%–100%`):** Real-time low-pass filtering, soundboard body EQ, and stereo hall impulse reverb let you dial in the exact brightness or softness of any preset while playing.

### 🛠️ Active Practice Suite (The Anti-YouTube Experience)
* **Wait Mode:** Freezes the falling notes right at the `NOW` line and waits until you press the highlighted keys on your connected MIDI keyboard (or touch screen) before advancing.
* **Hand Isolation (`L` / `R`):** Color-coded Left Hand (**Blue**) and Right Hand (**Green**) tracks can be muted and dimmed independently so you can practice one hand solo while the app plays the other.
* **Smart Left/Right Hand Voice Splitting:** Automatically separates single-track `.MID` files into playable Left and Right Hand parts using human hand-span constraints (10th interval limit) and voice-leading continuity.
* **A–B Section Looping with `+5%` Tempo Ramp:** Drag the orange `‹` and `›` handles on the timeline to loop tricky measures. Enable **+5% Speed Ramp** to start slow (`0.5x`) and automatically increase speed by `+0.05x` after each clean pass.
* **Plug-and-Play WebMIDI & `.MID` Loader:** Connect any USB or Bluetooth MIDI keyboard for live key detection and sustain pedal (`CC 64`) support, or click **+ Open .MID File** on the Home screen to load any standard `.mid` / `.midi` file.
* **Hybrid AI Audio-to-MIDI Transcriber ("Bring Your Own Song"):** Click **+ AI Audio Song** on the Home screen to convert any `.mp3`, `.wav`, `.m4a`, or `.ogg` audio file into playable, hand-separated falling piano notes:
  * **Zero-Dependency In-Browser Polyphonic Engine:** Uses Web Audio API spectral analysis with harmonic overtone suppression directly on your iPad or browser—working 100% offline on GitHub Pages.
  * **Optional Python FastAPI / Hugging Face ZeroGPU Backend (`backend/`):** Connect to a Hugging Face ZeroGPU Gradio Space or `localhost:8000` for `HT-Demucs` piano stem separation and neural polyphonic transcription (`ByteDance Piano Transcription` / `Basic Pitch` / `Librosa CQT`).
* **Persistent Custom Song Library & `.MID` Exporter:** Uploaded `.MID` files, AI-transcribed songs, and visual note edits are automatically saved in `IndexedDB` across sessions on your iPad, and any song can be exported as a two-track Standard `.MID` file.

### 🎓 Advanced Learning Aids & Cable-Free Practice (Phase 4B)
* **Real-Time Microphone Listen Mode (`🎤 Mic`):** Practice cable-free on any acoustic or speaker digital piano (such as a Casio CT-S200) on iPad without needing USB cables or WebMIDI. Features high-resolution Web Audio pitch detection with harmonic overtone rejection and auto-muting synth to prevent acoustic feedback loops.
* **Live Chord Symbol Detector Badge:** Displays active chords in real time (`C`, `G/B`, `Am7`, `Fmaj7`, `Ddim`) above the runway using polyphonic pitch-class analysis with slash-bass recognition. Toggleable via Settings Popover.
* **Audible Woodblock Metronome & 1-Bar Count-In (`1-2-3-4`):** Crisp woodblock downbeat and upbeat metronome clicks synced to song tempo. Optional 4-beat visual and audible count-in pre-roll before starting playback or loop wraps.
* **Live Hit Accuracy %, Streak Counter & Wrong-Note Red Flash:** Real-time feedback displaying practice accuracy percentage and consecutive hit streaks. Pressing incorrect keys triggers an instant high-visibility red key flash, red contact flare, and spark burst on the keyboard.
* **Editable Finger Number Badges (`1–5`):** Assign finger numbers (`1 = Thumb` .. `5 = Pinky`, `0 = Clear`) to notes in Edit Mode via the Note Inspector buttons or keyboard shortcuts `0`–`5`. Fingerings are displayed inside falling bars and persisted to `IndexedDB`.
* **Full Library Backup & Restore (`.keyflow.json`):** One-click export and import of your entire custom and edited song library from the Home Screen.

---

## 🎵 Built-In Song Library

KeyFlow ships with 5 built-in arrangements ready for immediate practice and visual customization:
1. **Let It Be** — *The Beatles*
2. **Imagine** — *John Lennon*
3. **Clair de Lune** — *Claude Debussy*
4. **Gymnopédie No. 1** — *Erik Satie*
5. **Pyramid Song** — *Radiohead*

---

## 🗺️ Roadmap & Changelog
* **Phase 1 (Complete):** Core 5s visualizer, auto-scaled keyboard, Mellow Felt Grand sound engine, A-B looping, Hand Isolation (L/R), Wait Mode, and `.MID` file upload.
* **Phase 2 (Complete):** Smart A-B Loop Tempo Ramp (+5%), Smart Left/Right hand voice-splitting algorithm, measure & beat gridlines.
* **Phase 3 (Complete):** Hybrid AI Audio-to-MIDI Transcription Pipeline (Hugging Face ZeroGPU / FastAPI backend + In-browser Web Audio polyphonic transcriber), persistent IndexedDB library, and standard `.MID` exporter.
* **Phase 4A (Complete):** Home Screen Song Library view, decluttered practice top bar, interactive Visual Note Editor on runway with 30-level Undo/Redo and group merge.
* **Phase 4B (Complete):** Real-Time Microphone Listen Mode (`🎤 Mic`), Live Chord Symbol Detector (`C`, `G/B`, `Am7`, `Fmaj7`), Audible Metronome & 1-Bar Count-In (`1-2-3-4`), Live Hit Accuracy % & Wrong-Note Red Flash, Editable Finger Number Badges (`1–5`), and Full Library `.keyflow.json` Backup & Restore.

---

## 📂 Project Architecture

KeyFlow uses a clean, modular vanilla JavaScript architecture with **zero external runtime dependencies** and **no build step required**—ensuring instant loading on GitHub Pages, iPad Safari, and embedded environments:

```text
Piano-Flow/
├── index.html               # Home library view, clean top bar, editor toolbar & AI Audio modal
├── manifest.json            # PWA Web App Manifest for fullscreen iPad landscape mode
├── css/
│   └── styles.css           # Dark glassmorphic UI styling, Home grid & Editor toolbar rules
├── js/
│   ├── songs.js             # Pitch/frequency helpers & 5 built-in song arrangements
│   ├── midi-parser.js       # Native binary .MID file parser, smart hand splitter & auto-scaler
│   ├── audio-transcriber.js # Hybrid AI + Web Audio polyphonic transcriber, IndexedDB & .MID exporter
│   ├── audio.js             # Web Audio Acoustic Grand (default) & Mellow Felt sound engine
│   └── app.js               # 60 FPS visualizer, Home library, Visual Note Editor & WebMIDI controller
├── backend/
│   ├── app.py               # Hugging Face ZeroGPU Gradio Space + FastAPI (/health, /api/transcribe)
│   ├── server.py            # Standalone local FastAPI server
│   ├── packages.txt         # Debian system packages (ffmpeg, libsndfile1) for HF Spaces
│   ├── requirements.txt     # Python backend dependencies
│   └── README.md            # Setup & iPad connection guide
└── .agents/
    └── rules/
        └── AGENTS.md        # Workspace architecture rules & phased development roadmap
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
