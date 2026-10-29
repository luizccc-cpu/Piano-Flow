"""
KeyFlow (Piano-Flow) — Phase 3 Hybrid AI Audio-to-MIDI Backend Server
=====================================================================
FastAPI service providing:
1. Optional HT-Demucs stem separation (`htdemucs` / `htdemucs_6s`) to isolate
   piano/accompaniment from vocals, drums, and bass in full-mix songs.
2. Multi-engine Polyphonic Piano Transcription:
   - Tier 1: ByteDance `piano_transcription_inference` (High-Res Onsets & Frames + Pedal)
   - Tier 2: Spotify `basic_pitch` (Neural polyphonic audio-to-MIDI)
   - Tier 3: Built-in `librosa` CQT + Harmonic-Percussive Polyphonic Peak Tracker
     (guaranteed fallback so the server always works out-of-the-box).
3. CORS enabled (`allow_origins=["*"]`) so iPad Safari / Standalone Web App
   hosted on GitHub Pages can connect over local Wi-Fi (`http://<LAN-IP>:8000`)
   or `http://localhost:8000`.
"""

import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB limit

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False

try:
    from piano_transcription_inference import PianoTranscription, sample_rate as BD_SAMPLE_RATE
    HAS_BYTEDANCE = True
except ImportError:
    HAS_BYTEDANCE = False

try:
    from basic_pitch.inference import predict as basic_pitch_predict
    HAS_BASIC_PITCH = True
except ImportError:
    HAS_BASIC_PITCH = False

HAS_DEMUCS = shutil.which("demucs") is not None


app = FastAPI(
    title="KeyFlow AI Audio-to-MIDI Transcription Server",
    version="3.0.0",
    description="HT-Demucs stem isolation + Polyphonic Piano Transcription for KeyFlow Virtual Piano",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_available_engines() -> List[str]:
    engines = []
    if HAS_DEMUCS:
        engines.append("HT-Demucs")
    if HAS_BYTEDANCE:
        engines.append("ByteDance-Piano")
    if HAS_BASIC_PITCH:
        engines.append("Basic-Pitch")
    if HAS_LIBROSA:
        engines.append("Librosa-CQT")
    return engines or ["Standard-MIDI"]


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": "3.0.0",
        "engines": get_available_engines(),
        "demucs_available": HAS_DEMUCS,
        "bytedance_available": HAS_BYTEDANCE,
        "basic_pitch_available": HAS_BASIC_PITCH,
        "librosa_available": HAS_LIBROSA,
    }


def isolate_piano_with_demucs(input_audio_path: Path, work_dir: Path) -> Path:
    """
    Runs HT-Demucs to strip drums, bass, and vocals when `demucs` is installed.
    Falls back to the original audio file if Demucs is unavailable or fails.
    """
    if not HAS_DEMUCS:
        return input_audio_path

    try:
        out_dir = work_dir / "demucs_out"
        cmd = [
            "demucs",
            "-n", "htdemucs",
            "--two-stems", "other",
            "-o", str(out_dir),
            str(input_audio_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        stem_dir = out_dir / "htdemucs" / input_audio_path.stem
        other_wav = stem_dir / "other.wav"
        if other_wav.exists():
            return other_wav
    except Exception as exc:
        print(f"[KeyFlow Backend] Demucs stem separation skipped: {exc}")

    return input_audio_path


def transcribe_with_bytedance(audio_path: Path, output_midi_path: Path) -> bool:
    if not HAS_BYTEDANCE or not HAS_LIBROSA:
        return False
    try:
        audio, _ = librosa.load(str(audio_path), sr=BD_SAMPLE_RATE, mono=True)
        transcriber = PianoTranscription(device="cpu", checkpoint_path=None)
        transcriber.transcribe(audio, str(output_midi_path))
        return output_midi_path.exists()
    except Exception as exc:
        print(f"[KeyFlow Backend] ByteDance transcription failed: {exc}")
        return False


def transcribe_with_basic_pitch(audio_path: Path, output_midi_path: Path, sensitivity: float) -> bool:
    if not HAS_BASIC_PITCH:
        return False
    try:
        onset_thresh = float(np.clip(0.65 - sensitivity * 0.35, 0.25, 0.70))
        frame_thresh = float(np.clip(0.45 - sensitivity * 0.22, 0.18, 0.55))
        _, midi_data, _ = basic_pitch_predict(
            str(audio_path),
            onset_threshold=onset_thresh,
            frame_threshold=frame_thresh,
            minimum_note_length=55.0,
            minimum_frequency=librosa.midi_to_hz(28) if HAS_LIBROSA else 41.2,
            maximum_frequency=librosa.midi_to_hz(96) if HAS_LIBROSA else 2093.0,
        )
        midi_data.write(str(output_midi_path))
        return output_midi_path.exists()
    except Exception as exc:
        print(f"[KeyFlow Backend] Basic Pitch transcription failed: {exc}")
        return False


def transcribe_with_librosa_cqt(audio_path: Path, sensitivity: float) -> List[Dict[str, Any]]:
    """
    Polyphonic Constant-Q Transform (CQT) + Harmonic Suppression Piano Transcriber.
    Works out-of-the-box with standard `librosa` + `numpy`.
    """
    if not HAS_LIBROSA:
        raise RuntimeError("librosa is required for audio analysis.")

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    y_harm, _ = librosa.effects.hpss(y, margin=(1.2, 2.0))

    hop_length = 512
    fmin = librosa.midi_to_hz(28)  # E1
    n_bins = 69                    # MIDI 28..96 (E1..C7)

    cqt_mag = np.abs(
        librosa.cqt(
            y_harm,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=12,
        )
    )

    # Suppress 2nd, 3rd, and 4th harmonic overtones (+12, +19, +24 semitones)
    cleaned = cqt_mag.copy()
    for b in range(n_bins):
        e0 = cqt_mag[b]
        if b + 12 < n_bins:
            cleaned[b + 12] = np.maximum(0.0, cleaned[b + 12] - e0 * 0.45)
        if b + 19 < n_bins:
            cleaned[b + 19] = np.maximum(0.0, cleaned[b + 19] - e0 * 0.26)
        if b + 24 < n_bins:
            cleaned[b + 24] = np.maximum(0.0, cleaned[b + 24] - e0 * 0.16)

    max_val = float(np.max(cleaned)) if cleaned.size > 0 else 1.0
    if max_val <= 1e-8:
        return []
    norm = cleaned / max_val

    sens = float(np.clip(sensitivity, 0.15, 0.90))
    act_thresh = 0.24 - (sens - 0.5) * 0.18
    max_poly = 5
    frame_sec = hop_length / float(sr)

    active: Dict[int, Dict[str, float]] = {}
    notes: List[Dict[str, Any]] = []

    num_frames = norm.shape[1]
    for f_idx in range(num_frames):
        t = f_idx * frame_sec
        col = norm[:, f_idx]
        peaks = []
        for b in range(1, n_bins - 1):
            val = float(col[b])
            if val >= act_thresh and val >= float(col[b - 1]) * 0.94 and val >= float(col[b + 1]) * 0.94:
                peaks.append((b + 28, val))
        peaks.sort(key=lambda x: x[1], reverse=True)
        chosen = dict(peaks[:max_poly])

        for midi_num in list(active.keys()):
            st = active[midi_num]
            cur_val = chosen.get(midi_num, 0.0)
            if cur_val < act_thresh * 0.55:
                dur = t - st["start"]
                if dur >= 0.065:
                    notes.append({
                        "midi": int(midi_num),
                        "time": round(float(st["start"]), 4),
                        "duration": round(float(min(4.5, dur)), 4),
                        "velocity": round(float(np.clip(0.38 + np.sqrt(st["peak"]) * 0.56, 0.35, 0.96)), 3),
                    })
                del active[midi_num]
            else:
                if cur_val > st["peak"]:
                    st["peak"] = cur_val

        for midi_num, val in chosen.items():
            if midi_num not in active:
                active[midi_num] = {"start": t, "peak": val}

    end_t = num_frames * frame_sec
    for midi_num, st in active.items():
        dur = end_t - st["start"]
        if dur >= 0.065:
            notes.append({
                "midi": int(midi_num),
                "time": round(float(st["start"]), 4),
                "duration": round(float(min(4.5, dur)), 4),
                "velocity": round(float(np.clip(0.38 + np.sqrt(st["peak"]) * 0.56, 0.35, 0.96)), 3),
            })

    notes.sort(key=lambda n: (n["time"], n["midi"]))
    return notes


def notes_to_midi_bytes(notes: List[Dict[str, Any]], bpm: float = 110.0) -> bytes:
    if not HAS_PRETTY_MIDI:
        raise RuntimeError("pretty_midi is not installed")
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    for n in notes:
        vel = int(np.clip(round(float(n.get("velocity", 0.78)) * 127), 28, 124))
        start = float(n["time"])
        end = start + max(0.06, float(n["duration"]))
        piano.notes.append(pretty_midi.Note(velocity=vel, pitch=int(n["midi"]), start=start, end=end))
    pm.instruments.append(piano)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


@app.post("/api/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    isolate_piano: str = Form("true"),
    sensitivity: float = Form(0.55),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing audio filename.")

    should_isolate = str(isolate_piano).lower() in ("true", "1", "yes")
    raw_suffix = Path(file.filename).suffix or ".mp3"
    suffix = re.sub(r"[^a-zA-Z0-9.]", "", raw_suffix)[:10] or ".mp3"

    with tempfile.TemporaryDirectory(prefix="keyflow_ai_") as tmp_dir:
        work_dir = Path(tmp_dir)
        input_path = work_dir / f"upload{suffix}"
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio file exceeds 50 MB limit.")
        input_path.write_bytes(raw_bytes)

        target_audio = isolate_piano_with_demucs(input_path, work_dir) if should_isolate else input_path
        output_midi = work_dir / "transcribed.mid"

        # 1. Try ByteDance High-Resolution Piano Transcription first
        if transcribe_with_bytedance(target_audio, output_midi):
            return Response(content=output_midi.read_bytes(), media_type="audio/midi")

        # 2. Try Spotify Basic Pitch second
        if transcribe_with_basic_pitch(target_audio, output_midi, sensitivity):
            return Response(content=output_midi.read_bytes(), media_type="audio/midi")

        # 3. Fallback to Librosa CQT Polyphonic Piano Transcriber
        notes = transcribe_with_librosa_cqt(target_audio, sensitivity)
        if not notes:
            raise HTTPException(status_code=422, detail="No pitched piano notes detected in audio file.")

        if HAS_PRETTY_MIDI:
            midi_bytes = notes_to_midi_bytes(notes)
            return Response(content=midi_bytes, media_type="audio/midi")

        return JSONResponse(
            content={
                "title": Path(file.filename).stem,
                "bpm": 110,
                "notes": notes,
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
