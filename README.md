# 🎹 KeyFlow (Piano-Flow) — 5-Second Horizon Virtual Piano

**KeyFlow** is an iPad-first, zero-dependency web application designed to make learning piano songs faster, clearer, and more intuitive than reading traditional sheet music or pausing YouTube falling-note videos.

Built for a landscape tablet resting on your piano's music stand, KeyFlow pairs a **constant 5-second lookahead runway**, **dynamic octave auto-scaling**, and **explicit note badges inside every falling bar** with a **multi-sampled Acoustic Grand Piano audio engine** and active practice tools.

---

## ✨ Key Features

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
* **Plug-and-Play WebMIDI & `.MID` Loader:** Connect any USB or Bluetooth MIDI keyboard for live key detection and sustain pedal (`CC 64`) support, or click **Open .MID** to load any standard `.mid` / `.midi` file directly from your device.
* **Hybrid AI Audio-to-MIDI Transcriber ("Bring Your Own Song"):** Click **AI Audio** in the top bar to convert any `.mp3`, `.wav`, `.m4a`, or `.ogg` audio file into playable, hand-separated falling piano notes:
  * **Zero-Dependency In-Browser Polyphonic Engine:** Uses Web Audio API spectral analysis with harmonic overtone suppression directly on your iPad or browser—working 100% offline on GitHub Pages.
  * **Optional Python FastAPI Studio AI Backend (`backend/`):** Connect over local Wi-Fi or `localhost:8000` for `HT-Demucs` piano stem separation and neural polyphonic transcription (`ByteDance Piano Transcription` / `Basic Pitch` / `Librosa CQT`).
* **Persistent Custom Song Library & `.MID` Exporter:** Uploaded `.MID` files and AI-transcribed songs are automatically saved in `IndexedDB` across sessions on your iPad, and any song can be exported as a two-track Standard `.MID` file from the settings menu.

---

## 🎵 Built-In Song Library

KeyFlow ships with 5 built-in arrangements ready for immediate practice:
1. **Let It Be** — *The Beatles*
2. **Imagine** — *John Lennon*
3. **Clair de Lune** — *Claude Debussy*
4. **Gymnopédie No. 1** — *Erik Satie*
5. **Pyramid Song** — *Radiohead*

---

## 📂 Project Architecture

KeyFlow uses a clean, modular vanilla JavaScript architecture with **zero external runtime dependencies** and **no build step required**—ensuring instant loading on GitHub Pages, iPad Safari, and embedded environments:

```text
Piano-Flow/
├── index.html               # App shell, HUD controls, AI Audio modal & transport bar
├── manifest.json            # PWA Web App Manifest for fullscreen iPad landscape mode
├── css/
│   └── styles.css           # Dark glassmorphic UI styling & responsive layout rules
├── js/
│   ├── songs.js             # Pitch/frequency helpers & 5 built-in song arrangements
│   ├── midi-parser.js       # Native binary .MID file parser, smart hand splitter & auto-scaler
│   ├── audio-transcriber.js # Hybrid AI + Web Audio polyphonic transcriber, IndexedDB & .MID exporter
│   ├── audio.js             # Web Audio Acoustic Grand (default) & Mellow Felt sound engine
│   └── app.js               # 60 FPS canvas visualizer, Wait Mode, A-B loop & WebMIDI controller
├── backend/
│   ├── server.py            # FastAPI backend (HT-Demucs + ByteDance / Basic Pitch / Librosa CQT)
│   ├── requirements.txt     # Python backend dependencies
│   └── README.md            # Setup & iPad Wi-Fi LAN connection guide
└── .agents/
    └── rules/
        └── AGENTS.md        # Workspace architecture rules & phased development roadmap
```

---

## 🚀 Getting Started

### 1. Run Locally
Because KeyFlow has no build step or bundler dependencies, you can run it with any static file server:

```bash
git clone https://github.com/luizccc-cpu/Piano-Flow.git
cd Piano-Flow
python3 -m http.server 8000
```
Then open `http://localhost:8000` in your browser.

### 2. Install as a Fullscreen App on iPad
1. Open your hosted **GitHub Pages** URL (`https://luizccc-cpu.github.io/Piano-Flow/`) in **Safari** on your iPad.
2. Tap the **Share** icon in the top bar and select **Add to Home Screen**.
3. Launch **KeyFlow** from your Home Screen—it will open in standalone landscape fullscreen mode without Safari address bars.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
