#!/usr/bin/env python3
"""
KeyFlow Pipeline Comparison Tool (A/B Testing CLI)
=================================================
Compares stem separation (HT-Demucs 6S vs BS-RoFormer),
stem preparation (Legacy filtering vs Minimal ~20 Hz high-pass),
and polyphonic transcription engines (ByteDance Piano AI vs Transkun).

Generates side-by-side MIDIs and saves isolated stem WAVs:
- A_legacy.mid               (htdemucs_6s + legacy prep + ByteDance)
- B_minimalprep.mid          (htdemucs_6s + minimal prep + ByteDance)
- C_roformer_bytedance.mid   (BS-RoFormer + minimal prep + ByteDance)
- D_roformer_transkun.mid    (BS-RoFormer + minimal prep + Transkun)

Usage:
    python backend/tools/compare_pipelines.py path/to/song.mp3
    python backend/tools/compare_pipelines.py path/to/song.mp3 --sensitivity 55
    python backend/tools/compare_pipelines.py path/to/song.mp3 --skip-demucs
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure backend directory is importable
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from server import (
    HAS_AUDIO_SEPARATOR,
    HAS_DEMUCS,
    HAS_BYTEDANCE,
    HAS_TRANSKUN,
    HAS_PRETTY_MIDI,
    isolate_piano_with_roformer,
    isolate_piano_with_htdemucs_6s,
    _legacy_clean_isolated_piano_audio,
    prepare_stem_for_transcription,
    transcribe_with_bytedance,
    transcribe_with_transkun,
    filter_midi_bleed_notes,
)


def analyze_midi(midi_path: Path) -> Dict[str, Any]:
    """Computes note count, low-velocity notes (<40), and pitch range."""
    if not midi_path.exists():
        return {"notes": 0, "vel_under_40": 0, "pitch_range": "N/A"}

    if not HAS_PRETTY_MIDI:
        size_b = midi_path.stat().st_size
        return {"notes": "N/A", "vel_under_40": "N/A", "pitch_range": f"{size_b} bytes"}

    try:
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(str(midi_path))
        all_notes = []
        for inst in pm.instruments:
            if not inst.is_drum:
                all_notes.extend(inst.notes)

        count = len(all_notes)
        vel_under_40 = sum(1 for n in all_notes if n.velocity < 40)
        if all_notes:
            min_p = min(n.pitch for n in all_notes)
            max_p = max(n.pitch for n in all_notes)
            pitch_range = (
                f"{pretty_midi.note_number_to_name(min_p)} ({min_p}) - "
                f"{pretty_midi.note_number_to_name(max_p)} ({max_p})"
            )
        else:
            pitch_range = "None"

        return {
            "notes": count,
            "vel_under_40": vel_under_40,
            "pitch_range": pitch_range,
        }
    except Exception as exc:
        return {"notes": "Error", "vel_under_40": "Error", "pitch_range": str(exc)[:30]}


def run_comparison(audio_path: Path, sensitivity: float = 55.0, skip_demucs: bool = False) -> None:
    if not audio_path.exists():
        print(f"Error: Input audio file does not exist: {audio_path}")
        sys.exit(1)

    out_dir = audio_path.parent / f"{audio_path.stem}_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print(f"KeyFlow Pipeline Comparison Tool")
    print(f"Input Audio  : {audio_path}")
    print(f"Output Dir   : {out_dir}")
    print(f"Sensitivity  : {sensitivity}")
    print(f"Skip Demucs  : {skip_demucs}")
    print("=" * 80)

    # 1. HT-Demucs Separation (run once and reused for Rows A & B)
    demucs_raw_stem: Optional[Path] = None
    demucs_legacy_stem: Optional[Path] = None
    demucs_minimal_stem: Optional[Path] = None

    if skip_demucs:
        print("\n[1/4] Skipping HT-Demucs 6S separation (--skip-demucs specified).")
    elif HAS_DEMUCS:
        print("\n[1/4] Running HT-Demucs 6S separation (once for Rows A & B)...")
        t0 = time.perf_counter()
        raw_demucs = isolate_piano_with_htdemucs_6s(audio_path, out_dir)
        t_demucs_sep = time.perf_counter() - t0
        if raw_demucs and raw_demucs.exists():
            demucs_raw_stem = out_dir / "htdemucs_6s_raw.wav"
            shutil.copyfile(raw_demucs, demucs_raw_stem)
            print(f"  -> Saved raw HT-Demucs stem in {t_demucs_sep:.2f}s: {demucs_raw_stem.name}")

            # Legacy prep
            t_leg_0 = time.perf_counter()
            legacy_prepped = _legacy_clean_isolated_piano_audio(demucs_raw_stem, out_dir)
            t_demucs_leg = time.perf_counter() - t_leg_0
            demucs_legacy_stem = out_dir / "htdemucs_6s_legacy_prep.wav"
            if legacy_prepped != demucs_legacy_stem and legacy_prepped.exists():
                shutil.copyfile(legacy_prepped, demucs_legacy_stem)
            print(f"  -> Saved legacy prepped stem in {t_demucs_leg:.2f}s: {demucs_legacy_stem.name}")

            # Minimal prep
            t_min_0 = time.perf_counter()
            minimal_prepped = prepare_stem_for_transcription(demucs_raw_stem, out_dir, mode="minimal")
            t_demucs_min = time.perf_counter() - t_min_0
            demucs_minimal_stem = out_dir / "htdemucs_6s_minimal_prep.wav"
            if minimal_prepped != demucs_minimal_stem and minimal_prepped.exists():
                shutil.copyfile(minimal_prepped, demucs_minimal_stem)
            print(f"  -> Saved minimal prepped stem in {t_demucs_min:.2f}s: {demucs_minimal_stem.name}")
        else:
            print(f"  [WARN] HT-Demucs separation failed or returned no piano stem ({t_demucs_sep:.2f}s).")
    else:
        print("\n[SKIP] HT-Demucs is not installed or available on PATH. Rows A & B will be skipped.")

    # 2. BS-RoFormer Separation (run once and reused for Rows C & D)
    roformer_raw_stem: Optional[Path] = None
    roformer_minimal_stem: Optional[Path] = None

    if HAS_AUDIO_SEPARATOR:
        print("\n[2/4] Running BS-RoFormer SW separation (once for Rows C & D)...")
        t0 = time.perf_counter()
        raw_roformer = isolate_piano_with_roformer(audio_path, out_dir)
        t_roformer_sep = time.perf_counter() - t0
        if raw_roformer and raw_roformer.exists():
            roformer_raw_stem = out_dir / "roformer_raw.wav"
            shutil.copyfile(raw_roformer, roformer_raw_stem)
            print(f"  -> Saved raw BS-RoFormer stem in {t_roformer_sep:.2f}s: {roformer_raw_stem.name}")

            # Minimal prep
            t_min_0 = time.perf_counter()
            minimal_prepped = prepare_stem_for_transcription(roformer_raw_stem, out_dir, mode="minimal")
            t_roformer_min = time.perf_counter() - t_min_0
            roformer_minimal_stem = out_dir / "roformer_minimal_prep.wav"
            if minimal_prepped != roformer_minimal_stem and minimal_prepped.exists():
                shutil.copyfile(minimal_prepped, roformer_minimal_stem)
            print(f"  -> Saved minimal prepped stem in {t_roformer_min:.2f}s: {roformer_minimal_stem.name}")
        else:
            print(f"  [WARN] BS-RoFormer separation failed or returned no piano stem ({t_roformer_sep:.2f}s).")
    else:
        print("\n[SKIP] audio-separator is not installed. Rows C & D will be skipped.")

    # 3. Transcribe each pipeline row
    results: List[Dict[str, Any]] = []

    # Row A: A_legacy.mid (htdemucs_6s + legacy + ByteDance)
    print("\n--- Running Row A (A_legacy.mid: htdemucs_6s | legacy | ByteDance) ---")
    midi_a = out_dir / "A_legacy.mid"
    if skip_demucs:
        print("  -> SKIPPED: --skip-demucs active.")
        results.append({"row": "A", "file": "A_legacy.mid", "status": "SKIPPED (--skip-demucs)", "time_s": "-"})
    elif not HAS_DEMUCS:
        print("  -> SKIPPED: HT-Demucs dependency not available.")
        results.append({"row": "A", "file": "A_legacy.mid", "status": "SKIPPED (missing Demucs)", "time_s": "-"})
    elif not HAS_BYTEDANCE:
        print("  -> SKIPPED: ByteDance transcriber dependency not available.")
        results.append({"row": "A", "file": "A_legacy.mid", "status": "SKIPPED (missing ByteDance)", "time_s": "-"})
    elif not demucs_legacy_stem or not demucs_legacy_stem.exists():
        print("  -> SKIPPED: HT-Demucs legacy stem unavailable.")
        results.append({"row": "A", "file": "A_legacy.mid", "status": "SKIPPED (Demucs failed)", "time_s": "-"})
    else:
        t0 = time.perf_counter()
        ok = transcribe_with_bytedance(demucs_legacy_stem, midi_a)
        if ok:
            filter_midi_bleed_notes(midi_a, sensitivity)
            elapsed = time.perf_counter() - t0
            stats = analyze_midi(midi_a)
            results.append({"row": "A", "file": "A_legacy.mid", "status": "DONE", "time_s": f"{elapsed:.2f}s", **stats})
            print(f"  -> Generated {midi_a.name} in {elapsed:.2f}s: {stats['notes']} notes, {stats['vel_under_40']} low-velocity, range {stats['pitch_range']}")
        else:
            elapsed = time.perf_counter() - t0
            results.append({"row": "A", "file": "A_legacy.mid", "status": "FAILED", "time_s": f"{elapsed:.2f}s"})
            print(f"  -> Failed to transcribe {midi_a.name} ({elapsed:.2f}s)")

    # Row B: B_minimalprep.mid (htdemucs_6s + minimal + ByteDance)
    print("\n--- Running Row B (B_minimalprep.mid: htdemucs_6s | minimal | ByteDance) ---")
    midi_b = out_dir / "B_minimalprep.mid"
    if skip_demucs:
        print("  -> SKIPPED: --skip-demucs active.")
        results.append({"row": "B", "file": "B_minimalprep.mid", "status": "SKIPPED (--skip-demucs)", "time_s": "-"})
    elif not HAS_DEMUCS:
        print("  -> SKIPPED: HT-Demucs dependency not available.")
        results.append({"row": "B", "file": "B_minimalprep.mid", "status": "SKIPPED (missing Demucs)", "time_s": "-"})
    elif not HAS_BYTEDANCE:
        print("  -> SKIPPED: ByteDance transcriber dependency not available.")
        results.append({"row": "B", "file": "B_minimalprep.mid", "status": "SKIPPED (missing ByteDance)", "time_s": "-"})
    elif not demucs_minimal_stem or not demucs_minimal_stem.exists():
        print("  -> SKIPPED: HT-Demucs minimal stem unavailable.")
        results.append({"row": "B", "file": "B_minimalprep.mid", "status": "SKIPPED (Demucs failed)", "time_s": "-"})
    else:
        t0 = time.perf_counter()
        ok = transcribe_with_bytedance(demucs_minimal_stem, midi_b)
        if ok:
            filter_midi_bleed_notes(midi_b, sensitivity)
            elapsed = time.perf_counter() - t0
            stats = analyze_midi(midi_b)
            results.append({"row": "B", "file": "B_minimalprep.mid", "status": "DONE", "time_s": f"{elapsed:.2f}s", **stats})
            print(f"  -> Generated {midi_b.name} in {elapsed:.2f}s: {stats['notes']} notes, {stats['vel_under_40']} low-velocity, range {stats['pitch_range']}")
        else:
            elapsed = time.perf_counter() - t0
            results.append({"row": "B", "file": "B_minimalprep.mid", "status": "FAILED", "time_s": f"{elapsed:.2f}s"})
            print(f"  -> Failed to transcribe {midi_b.name} ({elapsed:.2f}s)")

    # Row C: C_roformer_bytedance.mid (BS-RoFormer + minimal + ByteDance)
    print("\n--- Running Row C (C_roformer_bytedance.mid: BS-RoFormer | minimal | ByteDance) ---")
    midi_c = out_dir / "C_roformer_bytedance.mid"
    if not HAS_AUDIO_SEPARATOR:
        print("  -> SKIPPED: audio-separator dependency not available.")
        results.append({"row": "C", "file": "C_roformer_bytedance.mid", "status": "SKIPPED (missing audio-separator)", "time_s": "-"})
    elif not HAS_BYTEDANCE:
        print("  -> SKIPPED: ByteDance transcriber dependency not available.")
        results.append({"row": "C", "file": "C_roformer_bytedance.mid", "status": "SKIPPED (missing ByteDance)", "time_s": "-"})
    elif not roformer_minimal_stem or not roformer_minimal_stem.exists():
        print("  -> SKIPPED: BS-RoFormer minimal stem unavailable.")
        results.append({"row": "C", "file": "C_roformer_bytedance.mid", "status": "SKIPPED (RoFormer failed)", "time_s": "-"})
    else:
        t0 = time.perf_counter()
        ok = transcribe_with_bytedance(roformer_minimal_stem, midi_c)
        if ok:
            filter_midi_bleed_notes(midi_c, sensitivity)
            elapsed = time.perf_counter() - t0
            stats = analyze_midi(midi_c)
            results.append({"row": "C", "file": "C_roformer_bytedance.mid", "status": "DONE", "time_s": f"{elapsed:.2f}s", **stats})
            print(f"  -> Generated {midi_c.name} in {elapsed:.2f}s: {stats['notes']} notes, {stats['vel_under_40']} low-velocity, range {stats['pitch_range']}")
        else:
            elapsed = time.perf_counter() - t0
            results.append({"row": "C", "file": "C_roformer_bytedance.mid", "status": "FAILED", "time_s": f"{elapsed:.2f}s"})
            print(f"  -> Failed to transcribe {midi_c.name} ({elapsed:.2f}s)")

    # Row D: D_roformer_transkun.mid (BS-RoFormer + minimal + Transkun)
    print("\n--- Running Row D (D_roformer_transkun.mid: BS-RoFormer | minimal | Transkun) ---")
    midi_d = out_dir / "D_roformer_transkun.mid"
    if not HAS_AUDIO_SEPARATOR:
        print("  -> SKIPPED: audio-separator dependency not available.")
        results.append({"row": "D", "file": "D_roformer_transkun.mid", "status": "SKIPPED (missing audio-separator)", "time_s": "-"})
    elif not HAS_TRANSKUN:
        print("  -> SKIPPED: Transkun transcriber dependency not available.")
        results.append({"row": "D", "file": "D_roformer_transkun.mid", "status": "SKIPPED (missing Transkun)", "time_s": "-"})
    elif not roformer_minimal_stem or not roformer_minimal_stem.exists():
        print("  -> SKIPPED: BS-RoFormer minimal stem unavailable.")
        results.append({"row": "D", "file": "D_roformer_transkun.mid", "status": "SKIPPED (RoFormer failed)", "time_s": "-"})
    else:
        t0 = time.perf_counter()
        ok = transcribe_with_transkun(roformer_minimal_stem, midi_d)
        if ok:
            filter_midi_bleed_notes(midi_d, sensitivity)
            elapsed = time.perf_counter() - t0
            stats = analyze_midi(midi_d)
            results.append({"row": "D", "file": "D_roformer_transkun.mid", "status": "DONE", "time_s": f"{elapsed:.2f}s", **stats})
            print(f"  -> Generated {midi_d.name} in {elapsed:.2f}s: {stats['notes']} notes, {stats['vel_under_40']} low-velocity, range {stats['pitch_range']}")
        else:
            elapsed = time.perf_counter() - t0
            results.append({"row": "D", "file": "D_roformer_transkun.mid", "status": "FAILED", "time_s": f"{elapsed:.2f}s"})
            print(f"  -> Failed to transcribe {midi_d.name} ({elapsed:.2f}s)")

    # 4. Summary Table
    print("\n" + "=" * 105)
    print(f"{'Row':<4} | {'Output File':<25} | {'Separator':<13} | {'Prep':<8} | {'Transcriber':<10} | {'Time':<8} | {'Status':<16} | {'Notes':<6} | {'Vel<40':<6} | {'Pitch Range'}")
    print("-" * 105)
    meta = {
        "A": ("htdemucs_6s", "legacy", "ByteDance"),
        "B": ("htdemucs_6s", "minimal", "ByteDance"),
        "C": ("BS-RoFormer", "minimal", "ByteDance"),
        "D": ("BS-RoFormer", "minimal", "Transkun"),
    }
    for res in results:
        r = res["row"]
        sep, prep, tra = meta[r]
        status = res.get("status", "")
        time_str = res.get("time_s", "-")
        notes = str(res.get("notes", "-"))
        vel = str(res.get("vel_under_40", "-"))
        prange = str(res.get("pitch_range", "-"))
        print(f"{r:<4} | {res['file']:<25} | {sep:<13} | {prep:<8} | {tra:<10} | {time_str:<8} | {status:<16} | {notes:<6} | {vel:<6} | {prange}")
    print("=" * 105)
    print(f"Comparison artifacts saved to: {out_dir}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="KeyFlow Transcription Pipeline A/B Comparison CLI")
    parser.add_argument("audio_path", type=str, help="Path to input audio mix file")
    parser.add_argument("--sensitivity", type=float, default=55.0, help="Note onset sensitivity (default: 55)")
    parser.add_argument("--skip-demucs", action="store_true", help="Skip HT-Demucs separation and rows A/B to benchmark BS-RoFormer only")
    args = parser.parse_args()

    run_comparison(Path(args.audio_path), sensitivity=args.sensitivity, skip_demucs=args.skip_demucs)


if __name__ == "__main__":
    main()
