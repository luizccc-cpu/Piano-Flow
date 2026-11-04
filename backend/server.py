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

import base64
import io
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Tuple

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB limit
ALLOWED_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".webm", ".mp4"}
ALLOWED_OMR_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".musicxml", ".mxl", ".xml"}

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from scipy.signal import butter, sosfiltfilt
    HAS_SCIPY_SIGNAL = True
except ImportError:
    HAS_SCIPY_SIGNAL = False

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

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import scipy.ndimage as ndi
    HAS_NDI = True
except ImportError:
    HAS_NDI = False

try:
    import pypdfium2 as pdfium
    HAS_PDFIUM = True
except ImportError:
    HAS_PDFIUM = False

try:
    import music21
    HAS_MUSIC21 = True
except ImportError:
    HAS_MUSIC21 = False

HAS_DEMUCS = shutil.which("demucs") is not None

BYTEDANCE_CHECKPOINT_URL = (
    "https://zenodo.org/record/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
BYTEDANCE_CHECKPOINT_PATH = Path(tempfile.gettempdir()) / "piano_model.pth"
_checkpoint_lock = threading.Lock()


def ensure_bytedance_checkpoint() -> Path:
    """Thread-safely downloads the ByteDance Piano Transcription model weights to tempdir on first use."""
    if BYTEDANCE_CHECKPOINT_PATH.exists() and BYTEDANCE_CHECKPOINT_PATH.stat().st_size > 10_000_000:
        return BYTEDANCE_CHECKPOINT_PATH

    with _checkpoint_lock:
        if BYTEDANCE_CHECKPOINT_PATH.exists() and BYTEDANCE_CHECKPOINT_PATH.stat().st_size > 10_000_000:
            return BYTEDANCE_CHECKPOINT_PATH

        tmp_download = BYTEDANCE_CHECKPOINT_PATH.with_suffix(f".part_{os.getpid()}_{threading.get_ident()}")
        try:
            urllib.request.urlretrieve(BYTEDANCE_CHECKPOINT_URL, str(tmp_download))
            tmp_download.replace(BYTEDANCE_CHECKPOINT_PATH)
        finally:
            if tmp_download.exists():
                try:
                    tmp_download.unlink()
                except Exception:
                    pass
        return BYTEDANCE_CHECKPOINT_PATH


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
        engines.append("HT-Demucs-6S")
    if HAS_BYTEDANCE:
        engines.append("ByteDance-Piano")
    if HAS_BASIC_PITCH:
        engines.append("Basic-Pitch")
    if HAS_LIBROSA:
        engines.append("Librosa-CQT")
    engines.append("Sheet-OMR")
    return engines


@app.get("/health")
@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": "3.0.0",
        "engines": get_available_engines(),
        "demucs_available": HAS_DEMUCS,
        "bytedance_available": HAS_BYTEDANCE,
        "basic_pitch_available": HAS_BASIC_PITCH,
        "librosa_available": HAS_LIBROSA,
        "pdfium_available": HAS_PDFIUM,
        "omr_available": HAS_PIL and HAS_PRETTY_MIDI and HAS_NDI,
    }


def clean_isolated_piano_audio(stem_wav_path: Path, work_dir: Path) -> Path:
    """
    Applies post-Demucs harmonic and piano-range filtering:
    1. Loads audio at ByteDance sample rate (16 kHz mono).
    2. Strips residual drum/percussive transients via HPSS (margin 1.0 harm vs 2.2 perc).
    3. Bandpasses to piano fundamental range (27.5 Hz A0 to 4400 Hz C8).
    4. Normalizes peak amplitude and saves to piano_cleaned.wav.
    """
    if not HAS_LIBROSA:
        return stem_wav_path

    try:
        y, sr = librosa.load(str(stem_wav_path), sr=BD_SAMPLE_RATE, mono=True)
        if y.size == 0 or np.max(np.abs(y)) < 1e-6:
            return stem_wav_path

        # 1. HPSS to strip percussive bleed from drums/cymbals
        y_harm, _ = librosa.effects.hpss(y, margin=(1.0, 2.2))

        # 2. Bandpass filter for piano fundamental range (27.5 Hz - 4400 Hz)
        y_filtered = y_harm
        if HAS_SCIPY_SIGNAL and sr > 8800:
            try:
                nyq = 0.5 * sr
                low = 27.5 / nyq
                high = min(4400.0, nyq - 50.0) / nyq
                sos = butter(2, [low, high], btype="bandpass", output="sos")
                y_filtered = sosfiltfilt(sos, y_harm)
            except Exception as f_err:
                print(f"[KeyFlow Backend] Bandpass filter skipped: {f_err}")
                y_filtered = y_harm

        # 3. Peak amplitude normalization
        peak = float(np.max(np.abs(y_filtered)))
        if peak > 1e-6:
            y_norm = y_filtered * (0.95 / peak)
        else:
            y_norm = y_filtered

        cleaned_path = work_dir / "piano_cleaned.wav"
        if HAS_SOUNDFILE:
            sf.write(str(cleaned_path), y_norm.astype(np.float32), sr)
            if cleaned_path.exists() and cleaned_path.stat().st_size > 1000:
                return cleaned_path
    except Exception as exc:
        print(f"[KeyFlow Backend] Post-Demucs cleaning error: {exc}")

    return stem_wav_path


def isolate_piano_with_demucs(input_audio_path: Path, work_dir: Path) -> Path:
    """
    Isolates dedicated piano stem using 6-stem HT-Demucs (`htdemucs_6s --two-stems=piano`).
    Falls back to 4-stem `htdemucs --two-stems=other` and Librosa HPSS if unavailable.
    Cleans output stem with harmonic HPSS and piano-range bandpass filtering.
    """
    if not HAS_DEMUCS:
        return input_audio_path

    out_dir = work_dir / "demucs_out"
    device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

    # 1. Primary: 6-stem dedicated piano model (htdemucs_6s --two-stems piano)
    try:
        cmd_6s = [
            "demucs",
            "-n", "htdemucs_6s",
            "--two-stems", "piano",
            "-d", device,
            "-o", str(out_dir),
            str(input_audio_path),
        ]
        subprocess.run(cmd_6s, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        piano_wav = out_dir / "htdemucs_6s" / input_audio_path.stem / "piano.wav"
        if piano_wav.exists():
            return clean_isolated_piano_audio(piano_wav, work_dir)
    except Exception as exc:
        print(f"[KeyFlow Backend] htdemucs_6s dedicated piano isolation fallback: {exc}")
    finally:
        if HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # 2. Fallback: 4-stem model (htdemucs --two-stems other)
    try:
        cmd_4s = [
            "demucs",
            "-n", "htdemucs",
            "--two-stems", "other",
            "-d", device,
            "-o", str(out_dir),
            str(input_audio_path),
        ]
        subprocess.run(cmd_4s, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
        other_wav = out_dir / "htdemucs" / input_audio_path.stem / "other.wav"
        if other_wav.exists():
            return clean_isolated_piano_audio(other_wav, work_dir)
    except Exception as exc:
        print(f"[KeyFlow Backend] Demucs 4-stem fallback failed: {exc}")
    finally:
        if HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return input_audio_path


def transcribe_with_bytedance(audio_path: Path, output_midi_path: Path) -> bool:
    if not HAS_BYTEDANCE or not HAS_LIBROSA:
        return False
    try:
        ckpt_path = ensure_bytedance_checkpoint()
        audio, _ = librosa.load(str(audio_path), sr=BD_SAMPLE_RATE, mono=True)
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        transcriber = PianoTranscription(device=device, checkpoint_path=str(ckpt_path))
        transcriber.transcribe(audio, str(output_midi_path))
        return output_midi_path.exists() and output_midi_path.stat().st_size > 80
    except Exception as exc:
        print(f"[KeyFlow Backend] ByteDance transcription failed: {exc}")
        return False
    finally:
        if HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def transcribe_with_basic_pitch(audio_path: Path, output_midi_path: Path, sensitivity: float) -> bool:
    if not HAS_BASIC_PITCH:
        return False
    try:
        sens_norm = float(np.clip(sensitivity / 100.0 if sensitivity > 1.0 else sensitivity, 0.15, 0.90))
        onset_thresh = float(np.clip(0.65 - sens_norm * 0.35, 0.25, 0.70))
        frame_thresh = float(np.clip(0.45 - sens_norm * 0.22, 0.18, 0.55))
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


def filter_midi_bleed_notes(midi_path: Path, sensitivity: float = 55.0) -> bool:
    """
    Post-filters low-velocity bleed and micro-glitches from ByteDance transcription:
    1. Removes notes outside 88-key piano range (MIDI 21..108).
    2. Velocity gate mapped from sensitivity (15..85 -> min_vel 38..20; default 55 -> 28).
    3. Removes sub-55ms micro-glitches (duration < 0.055s).
    4. Caps simultaneous note onsets within any 30ms window to 8 notes (keeping highest velocity).
    """
    if not HAS_PRETTY_MIDI or not midi_path.exists():
        return False
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        sens_clamped = float(np.clip(sensitivity, 15.0, 85.0))
        min_vel = int(round(38.0 - ((sens_clamped - 15.0) / 70.0) * 18.0))
        min_vel = max(15, min(45, min_vel))

        total_kept = 0
        for inst in pm.instruments:
            if inst.is_drum:
                continue

            # Step 1: Filter pitch range, minimum velocity, and minimum duration
            valid_notes = []
            for n in inst.notes:
                dur = n.end - n.start
                if 21 <= n.pitch <= 108 and n.velocity >= min_vel and dur >= 0.055:
                    valid_notes.append(n)

            valid_notes.sort(key=lambda n: n.start)

            # Step 2: Cap simultaneous note onsets within 30ms window to 8 notes
            filtered_notes = []
            i = 0
            while i < len(valid_notes):
                w_start = valid_notes[i].start
                w_notes = []
                while i < len(valid_notes) and (valid_notes[i].start - w_start) <= 0.030:
                    w_notes.append(valid_notes[i])
                    i += 1
                if len(w_notes) > 8:
                    w_notes.sort(key=lambda n: n.velocity, reverse=True)
                    w_notes = w_notes[:8]
                filtered_notes.extend(w_notes)

            filtered_notes.sort(key=lambda n: (n.start, n.pitch))
            inst.notes = filtered_notes
            total_kept += len(filtered_notes)

        if total_kept > 0:
            pm.write(str(midi_path))
            return True
        return False
    except Exception as exc:
        print(f"[KeyFlow Filter] Error filtering MIDI bleed notes: {exc}")
        return False


def transcribe_with_librosa_cqt(audio_path: Path, sensitivity: float) -> Tuple[List[Dict[str, Any]], float]:
    """
    Polyphonic Constant-Q Transform (CQT) + Harmonic Suppression Piano Transcriber.
    Works out-of-the-box with standard `librosa` + `numpy`.
    """
    if not HAS_LIBROSA:
        raise RuntimeError("librosa is required for audio analysis.")

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    y_harm, _ = librosa.effects.hpss(y, margin=(1.2, 2.0))

    try:
        tempo_arr = librosa.beat.tempo(y=y_harm, sr=sr)
        raw_bpm = float(tempo_arr[0]) if hasattr(tempo_arr, "__getitem__") and len(tempo_arr) > 0 else float(tempo_arr)
        est_bpm = float(np.clip(raw_bpm, 60.0, 180.0))
    except Exception:
        est_bpm = 112.0

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
        return [], est_bpm
    norm = cleaned / max_val

    sens_norm = float(np.clip(sensitivity / 100.0 if sensitivity > 1.0 else sensitivity, 0.15, 0.90))
    act_thresh = 0.24 - (sens_norm - 0.5) * 0.18
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
    return notes, est_bpm


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
    sensitivity: str = Form("55"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing audio filename.")

    should_isolate = str(isolate_piano).lower() in ("true", "1", "yes")
    raw_suffix = Path(file.filename).suffix.lower()
    suffix = raw_suffix if raw_suffix in ALLOWED_SUFFIXES else ".mp3"
    clean_title = Path(file.filename).stem or "Transcribed Song"

    try:
        sens_val = float(sensitivity)
    except ValueError:
        sens_val = 55.0

    sens_norm = float(np.clip(sens_val / 100.0 if sens_val > 1.0 else sens_val, 0.15, 0.90))

    raw_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds 50 MB limit.")

    def _sync_transcribe():
        with tempfile.TemporaryDirectory(prefix="keyflow_ai_") as tmp_dir:
            work_dir = Path(tmp_dir)
            input_path = work_dir / f"upload{suffix}"
            input_path.write_bytes(raw_bytes)

            target_audio = isolate_piano_with_demucs(input_path, work_dir) if should_isolate else input_path
            output_midi = work_dir / "transcribed.mid"

            # 1. Try ByteDance High-Resolution Piano Transcription first
            if transcribe_with_bytedance(target_audio, output_midi):
                filter_midi_bleed_notes(output_midi, sens_val)
                return "midi", output_midi.read_bytes(), None

            # 2. Try Spotify Basic Pitch second
            if transcribe_with_basic_pitch(target_audio, output_midi, sens_norm):
                filter_midi_bleed_notes(output_midi, sens_val)
                return "midi", output_midi.read_bytes(), None

            # 3. Fallback to Librosa CQT Polyphonic Piano Transcriber
            notes, est_bpm = transcribe_with_librosa_cqt(target_audio, sens_norm)
            if not notes:
                raise HTTPException(status_code=422, detail="No pitched piano notes detected in audio file.")

            if HAS_PRETTY_MIDI:
                midi_bytes = notes_to_midi_bytes(notes, bpm=est_bpm)
                return "midi", midi_bytes, None

            return "json", None, {
                "title": clean_title,
                "bpm": round(est_bpm, 1),
                "notes": notes,
            }

    try:
        kind, data, json_payload = await run_in_threadpool(_sync_transcribe)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if kind == "midi":
        return Response(content=data, media_type="audio/midi")
    return JSONResponse(content=json_payload)


def render_pdf_to_images(pdf_path: Path, work_dir: Path, max_pages: int = 8) -> List[Path]:
    """Renders PDF pages to 200 DPI PNG images using pypdfium2."""
    if not HAS_PDFIUM:
        raise RuntimeError("pypdfium2 is required for PDF sheet music rendering.")
    pdf = pdfium.PdfDocument(str(pdf_path))
    page_paths = []
    num_pages = min(len(pdf), max_pages)
    for i in range(num_pages):
        page = pdf[i]
        image = page.render(scale=2.0).to_pil()
        out_path = work_dir / f"page_{i + 1}.png"
        image.save(str(out_path))
        page_paths.append(out_path)
    return page_paths


DIATONIC_PATTERN = [0, 2, 4, 5, 7, 9, 11]  # C, D, E, F, G, A, B


def diatonic_step_to_midi(base_step_in_c: int, base_octave: int, step_offset: int) -> int:
    """Maps diatonic line/space offsets on standard staves to MIDI pitch numbers."""
    target_step = base_step_in_c + step_offset
    octave_shift = target_step // 7
    note_in_oct = target_step % 7
    midi = (base_octave * 12) + 12 + (octave_shift * 12) + DIATONIC_PATTERN[note_in_oct]
    return int(np.clip(midi, 21, 108))


def transcribe_sheet_image_cv_to_midi(image_paths: List[Path], output_midi_path: Path) -> bool:
    """
    Deterministic Computer Vision Grand-Staff Visual OMR:
    1. Grayscale + adaptive binarization to detect sheet ink.
    2. Horizontal projection profile to detect 5-line staves & line spacing (d).
    3. Group staves into Grand Staff pairs (Top = Treble, Bottom = Bass).
    4. Remove thin staff lines via vertical morphology and locate notehead blobs.
    5. Map vertical line/space position to MIDI pitch and horizontal position to timeline.
    6. Writes Right Hand (Track 1) and Left Hand (Track 2) to Standard MIDI via pretty_midi.
    """
    if not HAS_PIL or not HAS_PRETTY_MIDI or not HAS_NDI:
        return False

    all_notes: List[Dict[str, Any]] = []
    current_time_offset = 0.0
    system_duration = 7.0  # seconds per grand staff system

    for img_path in image_paths:
        try:
            with Image.open(str(img_path)) as pil_img:
                gray = pil_img.convert("L")
                w, h = gray.size
                if w < 100 or h < 100:
                    continue
                if w > 2400:
                    scale = 2400.0 / w
                    gray = gray.resize((2400, int(round(h * scale))), Image.Resampling.BILINEAR)
                    w, h = gray.size
                img_arr = np.array(gray)
        except Exception as e:
            print(f"[KeyFlow OMR] Could not load image {img_path}: {e}")
            continue

        # 1. Binarize: black ink = 1, white paper = 0
        thresh = float(np.mean(img_arr)) * 0.85
        binary = (img_arr < thresh).astype(np.uint8)

        # 2. Horizontal projection profile
        h_proj = np.sum(binary, axis=1)
        line_thresh = max(int(w * 0.22), int(np.percentile(h_proj, 88)))

        line_ys = []
        in_run = False
        run_start = 0
        for y in range(h):
            if h_proj[y] >= line_thresh:
                if not in_run:
                    in_run = True
                    run_start = y
            else:
                if in_run:
                    in_run = False
                    line_ys.append(int(round((run_start + y - 1) / 2.0)))
        if in_run:
            line_ys.append(int(round((run_start + h - 1) / 2.0)))

        if len(line_ys) < 5:
            continue

        # 3. Group lines into 5-line staves
        detected_staves = []
        i = 0
        while i <= len(line_ys) - 5:
            spacings = [line_ys[i + j + 1] - line_ys[i + j] for j in range(4)]
            avg_d = np.mean(spacings)
            std_d = np.std(spacings)
            if 6.0 <= avg_d <= 45.0 and std_d <= 3.5:
                staff_lines = line_ys[i:i + 5]
                detected_staves.append({
                    "lines": staff_lines,
                    "top": staff_lines[0],
                    "bottom": staff_lines[4],
                    "d": float(avg_d),
                    "center": float(np.mean(staff_lines))
                })
                i += 5
            else:
                i += 1

        if not detected_staves:
            continue

        # 4. Group into grand staff pairs (Top = Treble, Bottom = Bass)
        grand_staves = []
        s_idx = 0
        while s_idx < len(detected_staves):
            st1 = detected_staves[s_idx]
            if s_idx + 1 < len(detected_staves):
                st2 = detected_staves[s_idx + 1]
                gap = st2["top"] - st1["bottom"]
                if 1.5 * st1["d"] <= gap <= 10.0 * st1["d"]:
                    grand_staves.append({"treble": st1, "bass": st2, "d": (st1["d"] + st2["d"]) / 2.0})
                    s_idx += 2
                    continue
            grand_staves.append({"treble": st1, "bass": None, "d": st1["d"]})
            s_idx += 1

        # 5. Notehead segmentation: vertical morphological opening
        typical_d = int(round(np.median([gs["d"] for gs in grand_staves])))
        v_kernel_h = max(3, int(round(typical_d * 0.55)))
        v_kernel = np.ones((v_kernel_h, 1), dtype=np.uint8)
        note_mask = ndi.binary_opening(binary, structure=v_kernel)

        labeled, num_features = ndi.label(note_mask)
        objects = ndi.find_objects(labeled)

        # 6. Map notehead blobs to pitch and time
        for gs_idx, gs in enumerate(grand_staves):
            d = gs["d"]
            treble = gs["treble"]
            bass = gs["bass"]
            sys_time_start = current_time_offset + (gs_idx * system_duration)

            min_y = treble["top"] - int(d * 4.0)
            max_y = (bass["bottom"] if bass else treble["bottom"]) + int(d * 4.0)
            x_start_margin = int(w * 0.08)
            x_end_margin = int(w * 0.96)
            sys_width = max(100, x_end_margin - x_start_margin)

            for sl_y, sl_x in objects:
                if sl_y is None or sl_x is None:
                    continue
                y_min, y_max = sl_y.start, sl_y.stop
                x_min, x_max = sl_x.start, sl_x.stop

                yc = (y_min + y_max) / 2.0
                xc = (x_min + x_max) / 2.0

                if not (min_y <= yc <= max_y and x_start_margin <= xc <= x_end_margin):
                    continue

                blob_h = y_max - y_min
                blob_w = x_max - x_min

                if not (0.55 * d <= blob_h <= 1.9 * d and 0.55 * d <= blob_w <= 2.6 * d):
                    continue
                if not (0.35 <= (blob_w / max(1.0, blob_h)) <= 3.2):
                    continue

                hand = 'R'
                if bass is not None:
                    dist_to_treble = abs(yc - treble["center"])
                    dist_to_bass = abs(yc - bass["center"])
                    if dist_to_bass < dist_to_treble:
                        hand = 'L'

                if hand == 'R':
                    step_offset = int(round((treble["bottom"] - yc) / (d / 2.0)))
                    midi = diatonic_step_to_midi(base_step_in_c=2, base_octave=4, step_offset=step_offset)
                else:
                    ref_bottom = bass["bottom"] if bass else treble["bottom"]
                    step_offset = int(round((ref_bottom - yc) / (d / 2.0)))
                    midi = diatonic_step_to_midi(base_step_in_c=4, base_octave=2, step_offset=step_offset)

                t = sys_time_start + ((xc - x_start_margin) / float(sys_width)) * (system_duration * 0.92)
                all_notes.append({
                    "midi": midi,
                    "time": round(float(t), 3),
                    "duration": 0.48,
                    "hand": hand,
                    "velocity": 85 if hand == 'R' else 78
                })

        current_time_offset += len(grand_staves) * system_duration

    if not all_notes:
        return False

    all_notes.sort(key=lambda n: (n["time"], n["midi"]))

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    rh_inst = pretty_midi.Instrument(program=0, name="Right Hand")
    lh_inst = pretty_midi.Instrument(program=0, name="Left Hand")

    for n in all_notes:
        note_obj = pretty_midi.Note(
            velocity=n["velocity"],
            pitch=n["midi"],
            start=n["time"],
            end=n["time"] + n["duration"]
        )
        if n["hand"] == 'L':
            lh_inst.notes.append(note_obj)
        else:
            rh_inst.notes.append(note_obj)

    pm.instruments.append(rh_inst)
    pm.instruments.append(lh_inst)
    pm.write(str(output_midi_path))
    return output_midi_path.exists() and output_midi_path.stat().st_size > 80


def extract_musicxml_text_from_mxl(mxl_bytes: bytes) -> str:
    """Extracts score XML text from a compressed .mxl ZIP archive."""
    buf = io.BytesIO(mxl_bytes)
    with zipfile.ZipFile(buf, "r") as z:
        target_name = None
        for name in z.namelist():
            if name.lower() == "meta-inf/container.xml":
                try:
                    c_xml = z.read(name).decode("utf-8", errors="ignore")
                    m = re.search(r'full-path\s*=\s*["\']([^"\']+)["\']', c_xml, re.IGNORECASE)
                    if m:
                        target_name = m.group(1).strip()
                except Exception:
                    pass

        if target_name and target_name in z.namelist():
            return z.read(target_name).decode("utf-8", errors="ignore")

        candidates = [n for n in z.namelist() if (n.lower().endswith(".musicxml") or n.lower().endswith(".xml")) and not n.lower().startswith("meta-inf/")]
        if not candidates:
            raise ValueError("No score XML file found in .mxl archive.")
        candidates.sort(key=lambda n: z.getinfo(n).file_size, reverse=True)
        return z.read(candidates[0]).decode("utf-8", errors="ignore")


@app.post("/api/omr")
async def omr_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Phase 5: Optical Music Recognition (OMR) & Sheet Music Endpoint.
    Accepts .pdf, .png, .jpg, .jpeg, .webp, .musicxml, or .mxl and returns
    either parsed MusicXML or base64 Standard MIDI.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing sheet music filename.")

    clean_title = Path(file.filename).stem or "Sheet Music Song"
    raw_suffix = Path(file.filename).suffix.lower()
    if raw_suffix not in ALLOWED_OMR_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported sheet music file extension: {raw_suffix}")

    raw_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded sheet music file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Sheet music file exceeds 50 MB limit.")

    # 1. Digital MusicXML or MXL uploads: extract and return MusicXML directly
    if raw_suffix in (".musicxml", ".xml"):
        return {
            "musicxml": raw_bytes.decode("utf-8", errors="ignore"),
            "title": clean_title,
        }

    if raw_suffix == ".mxl":
        try:
            xml_text = extract_musicxml_text_from_mxl(raw_bytes)
            return {
                "musicxml": xml_text,
                "title": clean_title,
            }
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not unpack .mxl archive: {exc}")

    # 2. Visual scans (.pdf, .png, .jpg, .jpeg, .webp): run OMR pipeline
    def _sync_omr() -> str:
        os.makedirs("/tmp", exist_ok=True)
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="keyflow_omr_") as tmp_dir:
            work_dir = Path(tmp_dir)
            input_path = work_dir / f"upload{raw_suffix}"
            input_path.write_bytes(raw_bytes)

            image_paths = []
            if raw_suffix == ".pdf":
                image_paths = render_pdf_to_images(input_path, work_dir, max_pages=8)
            else:
                image_paths = [input_path]

            if not image_paths:
                raise RuntimeError("No rendered pages available for OMR processing.")

            output_midi = work_dir / "transcribed_sheet.mid"
            ok = transcribe_sheet_image_cv_to_midi(image_paths, output_midi)
            if not ok or not output_midi.exists():
                raise RuntimeError("No readable piano staves or noteheads detected in sheet music image.")

            return base64.b64encode(output_midi.read_bytes()).decode("ascii")

    try:
        b64_str = await run_in_threadpool(_sync_omr)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "midi_base64": b64_str,
        "title": clean_title,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
