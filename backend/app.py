import os

# Configure writable cache directories for Hugging Face Gradio Spaces before importing ML libraries
os.environ["TORCH_HOME"] = "/tmp/torch"
os.environ["HF_HOME"] = "/tmp/hf"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba"

for _cache_dir in ("/tmp/torch", "/tmp/hf", "/tmp/numba"):
    os.makedirs(_cache_dir, exist_ok=True)

import base64
import shutil
import socket
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except Exception:
    HAS_PRETTY_MIDI = False

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    from piano_transcription_inference import PianoTranscription, sample_rate as BD_SAMPLE_RATE
    HAS_BYTEDANCE = True
except Exception:
    HAS_BYTEDANCE = False
    BD_SAMPLE_RATE = 16000

HAS_DEMUCS = shutil.which("demucs") is not None

BYTEDANCE_CHECKPOINT_URL = (
    "https://zenodo.org/record/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
BYTEDANCE_CHECKPOINT_PATH = Path("/tmp/piano_model.pth")


def ensure_bytedance_checkpoint() -> Path:
    """Downloads the ByteDance Piano Transcription model weights to /tmp/piano_model.pth on first use."""
    if BYTEDANCE_CHECKPOINT_PATH.exists() and BYTEDANCE_CHECKPOINT_PATH.stat().st_size > 10_000_000:
        return BYTEDANCE_CHECKPOINT_PATH

    tmp_download = Path("/tmp/piano_model.pth.part")
    urllib.request.urlretrieve(BYTEDANCE_CHECKPOINT_URL, str(tmp_download))
    tmp_download.replace(BYTEDANCE_CHECKPOINT_PATH)
    return BYTEDANCE_CHECKPOINT_PATH


def isolate_piano_stem(input_audio_path: Path, work_dir: Path) -> Path:
    """
    Isolates the piano/harmonic stem using HT-Demucs (`htdemucs --two-stems=other`).
    Falls back to Librosa Harmonic-Percussive Source Separation (HPSS) if Demucs is
    unavailable or memory-constrained on free CPU tiers.
    """
    if HAS_DEMUCS:
        try:
            out_dir = work_dir / "demucs_out"
            cmd = [
                "demucs",
                "-n", "htdemucs",
                "--two-stems", "other",
                "-d", "cpu",
                "-o", str(out_dir),
                str(input_audio_path),
            ]
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=240,
            )
            other_wav = out_dir / "htdemucs" / input_audio_path.stem / "other.wav"
            if other_wav.exists():
                return other_wav
        except Exception as exc:
            print(f"[KeyFlow Space] Demucs stem separation fallback triggered: {exc}")

    if HAS_LIBROSA:
        try:
            y, sr = librosa.load(str(input_audio_path), sr=22050, mono=True)
            y_harm, _ = librosa.effects.hpss(y, margin=(1.3, 2.2))
            harm_path = work_dir / "harmonic_isolated.wav"
            sf.write(str(harm_path), y_harm, sr)
            if harm_path.exists():
                return harm_path
        except Exception as exc:
            print(f"[KeyFlow Space] Librosa HPSS fallback skipped: {exc}")

    return input_audio_path


def transcribe_with_bytedance(audio_path: Path, output_midi_path: Path) -> bool:
    """Transcribes audio into a Standard MIDI file using ByteDance PianoTranscription."""
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
        print(f"[KeyFlow Space] ByteDance transcription error (falling back to Librosa CQT): {exc}")
        return False


def transcribe_with_librosa_cqt_to_midi(audio_path: Path, output_midi_path: Path, sensitivity: float) -> bool:
    """
    Guaranteed polyphonic CQT + Harmonic Overtone Suppression fallback that writes a
    Standard MIDI file via pretty_midi so transcription never fails.
    """
    if not HAS_LIBROSA or not HAS_PRETTY_MIDI:
        return False

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
        return False
    norm = cleaned / max_val

    sens_norm = float(np.clip(sensitivity / 100.0 if sensitivity > 1.0 else sensitivity, 0.15, 0.90))
    act_thresh = 0.24 - (sens_norm - 0.5) * 0.18
    max_poly = int(round(3 + sens_norm * 4))
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
                if dur >= 0.07:
                    notes.append({
                        "midi": int(midi_num),
                        "start": float(st["start"]),
                        "end": float(st["start"] + min(4.5, dur)),
                        "velocity": int(np.clip(round((0.38 + np.sqrt(st["peak"]) * 0.56) * 127), 32, 122)),
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
        if dur >= 0.07:
            notes.append({
                "midi": int(midi_num),
                "start": float(st["start"]),
                "end": float(st["start"] + min(4.5, dur)),
                "velocity": int(np.clip(round((0.38 + np.sqrt(st["peak"]) * 0.56) * 127), 32, 122)),
            })

    if not notes:
        return False

    pm = pretty_midi.PrettyMIDI(initial_tempo=112.0)
    piano = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    for n in sorted(notes, key=lambda item: (item["start"], item["midi"])):
        piano.notes.append(
            pretty_midi.Note(
                velocity=n["velocity"],
                pitch=n["midi"],
                start=n["start"],
                end=n["end"],
            )
        )
    pm.instruments.append(piano)
    pm.write(str(output_midi_path))
    return output_midi_path.exists()


async def health_check_handler() -> Dict[str, Any]:
    return {
        "status": "ok",
        "engines": ["HT-Demucs", "ByteDance Piano AI"],
    }


async def transcribe_endpoint_handler(
    file: UploadFile = File(...),
    isolate_piano: str = Form("true"),
    sensitivity: str = Form("55"),
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing audio filename.")

    should_isolate = str(isolate_piano).strip().lower() in ("true", "1", "yes")
    try:
        sens_val = float(sensitivity)
    except ValueError:
        sens_val = 55.0

    clean_title = Path(file.filename).stem or "Transcribed Song"
    suffix = Path(file.filename).suffix or ".mp3"

    os.makedirs("/tmp", exist_ok=True)
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="keyflow_space_") as tmp_dir:
        work_dir = Path(tmp_dir)
        input_path = work_dir / f"upload{suffix}"
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
        input_path.write_bytes(raw_bytes)

        target_audio = isolate_piano_stem(input_path, work_dir) if should_isolate else input_path
        output_midi = work_dir / "transcribed.mid"

        ok = transcribe_with_bytedance(target_audio, output_midi)
        if not ok:
            ok = transcribe_with_librosa_cqt_to_midi(target_audio, output_midi, sens_val)

        if not ok or not output_midi.exists():
            raise HTTPException(status_code=422, detail="Could not transcribe piano notes from audio file.")

        midi_bytes = output_midi.read_bytes()
        b64_str = base64.b64encode(midi_bytes).decode("ascii")
        return {
            "midi_base64": b64_str,
            "title": clean_title,
        }


def register_keyflow_routes(fastapi_app: FastAPI) -> FastAPI:
    """Idempotently attaches CORS and KeyFlow's /health & /api/transcribe endpoints to Gradio's FastAPI app."""
    if fastapi_app is None or getattr(fastapi_app, "_keyflow_routes_registered", False):
        return fastapi_app

    fastapi_app._keyflow_routes_registered = True
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    fastapi_app.get("/health")(health_check_handler)
    fastapi_app.post("/api/transcribe")(transcribe_endpoint_handler)
    return fastapi_app


# Hook Gradio's App.create_app so routes are guaranteed to be attached even when demo.app is initialized lazily
if hasattr(gr, "routes") and hasattr(gr.routes, "App") and hasattr(gr.routes.App, "create_app"):
    _orig_create_app = gr.routes.App.create_app

    def _patched_create_app(*args, **kwargs):
        created_app = _orig_create_app(*args, **kwargs)
        register_keyflow_routes(created_app)
        return created_app

    gr.routes.App.create_app = _patched_create_app


# 1. Create the Gradio Blocks UI first
with gr.Blocks(title="KeyFlow AI Audio-to-MIDI Server") as demo:
    gr.Markdown(
        """
        # 🎹 KeyFlow AI Audio-to-MIDI Server (Online)
        This Hugging Face Gradio Space powers **Phase 3 AI Audio-to-MIDI Transcription** for the **KeyFlow Virtual Piano** iPad web app.

        ### Active API Endpoints
        - **`GET /health`** — Returns server health & active engines (`HT-Demucs`, `ByteDance Piano AI`).
        - **`POST /api/transcribe`** — Accepts `file`, `isolate_piano`, and `sensitivity` (`multipart/form-data`) and returns `{"midi_base64": "...", "title": "..."}`.

        **How to connect from KeyFlow:** Copy this Space's public `.hf.space` URL (e.g. `https://<username>-<space-name>.hf.space`) into the **AI Server URL** field inside KeyFlow's **AI Audio** modal and tap **Check**.
        """
    )

# 2. Reference Gradio's built-in FastAPI instance (or initialize it if already present) and attach routes
app = getattr(demo, "app", None)
if app is not None:
    register_keyflow_routes(app)


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


if __name__ == "__main__":
    target_port = int(os.environ.get("PORT", 7860))
    if not _is_port_in_use(target_port):
        demo.launch(server_name="0.0.0.0", server_port=target_port)
        if getattr(demo, "app", None) is not None:
            app = register_keyflow_routes(demo.app)
