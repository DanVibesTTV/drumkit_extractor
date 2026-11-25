#!/usr/bin/env python3
"""
Drumkit Extractor
=================
Analyzes an audio file, finds every drum hit, classifies the hits
(kick, snare, hi-hats, toms, cymbals, ...) and exports the cleanest
examples of each element as individual WAV samples.

Pipeline
--------
1. Decode any audio format via ffmpeg (wav, mp3, flac, m4a, ogg, ...).
2. Isolate the drums: with Demucs if installed (best for full songs),
   otherwise a built-in harmonic/percussive separation (HPSS) that strips
   sustained bass, pads and vocals. Drum-only recordings are detected and
   left untouched.
3. Onset detection (multi-band spectral flux + adaptive threshold).
4. Feature extraction per hit (band energies, centroid, flatness, decay,
   fundamental pitch for toms).
5. Rule-based classification + pitch clustering for toms.
6. Ranking each hit by "cleanliness" (isolation from neighbouring hits,
   class confidence, loudness) and exporting the best N per element.

Usage
-----
    python drumkit_extractor.py song.mp3                 # -> song_drumkit/
    python drumkit_extractor.py song.mp3 -o kit -n 5     # 5 samples per element
    python drumkit_extractor.py drums.wav --no-separate  # already drums only
    python drumkit_extractor.py gui                      # simple desktop GUI

Requirements: numpy, scipy, ffmpeg on PATH.
Optional:     demucs (pip install demucs) for drum separation from full mixes,
              matplotlib for the overview image.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy import signal

SR = 44100

# ----------------------------------------------------------------------------
# Packaged (.exe) support
# ----------------------------------------------------------------------------

FROZEN = getattr(sys, "frozen", False)
# Folder with bundled files: PyInstaller's unpack dir, or the folder of this script.
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
# Windows GUI builds have no console: give libraries a harmless stdout/stderr.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
# Bundled Demucs model weights (full build) -> tell torch where to find them.
if (BUNDLE_DIR / "models").is_dir():
    os.environ.setdefault("TORCH_HOME", str(BUNDLE_DIR / "models"))


def find_ffmpeg() -> str | None:
    """Bundled ffmpeg first (next to the .exe / in the bundle), then PATH."""
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for d in (BUNDLE_DIR, BUNDLE_DIR / "ffmpeg", Path(sys.executable).parent):
        if (d / exe).is_file():
            return str(d / exe)
    return shutil.which("ffmpeg")


def _no_window() -> dict:
    """Keep ffmpeg from flashing a console window on Windows."""
    if os.name == "nt":
        return {"creationflags": 0x08000000}  # CREATE_NO_WINDOW
    return {}

# ----------------------------------------------------------------------------
# Audio I/O
# ----------------------------------------------------------------------------

def load_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """Decode any file ffmpeg understands to float32 stereo (2, n)."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found - install it or put ffmpeg.exe next to the program.")
    cmd = [ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le",
           "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr), "-"]
    r = subprocess.run(cmd, capture_output=True, **_no_window())
    if r.returncode != 0:
        raise RuntimeError(f"Could not decode {path}: {r.stderr.decode(errors='ignore')[-400:]}")
    raw = r.stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()
    if audio.shape[1] == 0:
        raise RuntimeError(f"No audio decoded from {path}")
    return audio


def write_wav(path: Path, audio: np.ndarray, sr: int = SR, bits: int = 24):
    """Write (channels, n) float audio as 16- or 24-bit PCM WAV."""
    import wave
    a = np.atleast_2d(audio)
    a = np.clip(a, -1.0, 1.0).T  # (n, ch)
    if bits == 16:
        data = (a * 32767.0).astype("<i2").tobytes()
        width = 2
    else:
        ints = (a * 8388607.0).astype("<i4")
        b = ints.reshape(-1, 1).view(np.uint8).reshape(-1, 4)[:, :3]
        data = b.tobytes()
        width = 3
    with wave.open(str(path), "wb") as w:
        w.setnchannels(a.shape[1])
        w.setsampwidth(width)
        w.setframerate(sr)
        w.writeframes(data)


# ----------------------------------------------------------------------------
# Drum separation (optional, Demucs)
# ----------------------------------------------------------------------------

def demucs_available() -> bool:
    try:
        import demucs  # noqa: F401
        return True
    except Exception:
        return False


def separate_drums(path: Path, log=print, model: str = "htdemucs") -> np.ndarray | None:
    """Return the drum stem as (2, n) at SR, or None if Demucs is unavailable."""
    if not demucs_available():
        return None
    log(f"Separating drums with Demucs ({model}) - this can take a minute...")
    with tempfile.TemporaryDirectory() as tmp:
        # Run Demucs in-process (works inside a packaged .exe too).
        # Decode with our ffmpeg first so Demucs gets a plain WAV.
        wav_in = Path(tmp) / "input.wav"
        write_wav(wav_in, load_audio(path), bits=16)
        try:
            from demucs.separate import main as demucs_main
            demucs_main(["--two-stems", "drums", "-n", model, "-o", tmp, str(wav_in)])
        except SystemExit as e:
            if e.code not in (0, None):
                log(f"Demucs failed (exit {e.code}), continuing with the full mix.")
                return None
        except Exception as e:
            log(f"Demucs failed ({e}), continuing with the full mix.")
            return None
        hits = list(Path(tmp).rglob("drums.wav"))
        if not hits:
            log("Demucs produced no drums stem, continuing with the full mix.")
            return None
        return load_audio(hits[0])


# ----------------------------------------------------------------------------
# Built-in fallback: harmonic / percussive separation (no extra installs)
# ----------------------------------------------------------------------------

def hpss(audio: np.ndarray, kernel_t: float = 0.25, kernel_f: float = 250.0,
         power: float = 2.0) -> tuple[np.ndarray, float]:
    """Median-filter HPSS (Fitzgerald 2010). Returns (percussive (ch, n),
    "mix-ness" 0..1: how much non-drum sound keeps playing between the hits). Sustained bass, pads, vocals and
    chords end up in the harmonic part; drum transients in the percussive part."""
    from scipy.ndimage import median_filter
    nfft, hop = 4096, 1024
    kt = max(3, int(kernel_t * SR / hop) | 1)
    kf = max(3, int(kernel_f / (SR / nfft)) | 1)
    out = np.zeros_like(audio)
    h_e = p_e = 0.0
    for ch in range(audio.shape[0]):
        _, _, Z = signal.stft(audio[ch], fs=SR, nperseg=nfft, noverlap=nfft - hop)
        M = np.abs(Z)
        H = median_filter(M, size=(1, kt), mode="nearest")
        P = median_filter(M, size=(kf, 1), mode="nearest")
        mask = P ** power / (H ** power + P ** power + 1e-20)
        h_e += float(np.sum((M * (1 - mask)) ** 2))
        p_e += float(np.sum((M * mask) ** 2))
        _, x = signal.istft(Z * mask, fs=SR, nperseg=nfft, noverlap=nfft - hop)
        n = min(len(x), audio.shape[1])
        out[ch, :n] = x[:n]
    # Drum-only recordings: the harmonic part is just ringing that decays after
    # each hit. Full mixes: bass/pads/vocals keep the harmonic floor up.
    harm = (audio - out).mean(axis=0)
    w = int(0.05 * SR)
    nfr = len(harm) // w
    if nfr < 4:
        return out, 0.0
    eh = np.sqrt((harm[:nfr * w].reshape(nfr, w) ** 2).mean(axis=1))
    ep = np.sqrt((out.mean(axis=0)[:nfr * w].reshape(nfr, w) ** 2).mean(axis=1))
    act = ep > 0.05 * ep.max()
    if act.sum() < 4:
        return out, 0.0
    floor = np.percentile(eh[act], 20) / (np.percentile(eh[act], 95) + 1e-12)
    return out, float(floor)


# ----------------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------------

N_FFT = 2048
HOP = 256

BANDS = {               # Hz ranges used for features
    "sub":   (20, 90),
    "low":   (90, 200),
    "lowmid": (200, 500),
    "mid":   (500, 2000),
    "high":  (2000, 6000),
    "air":   (6000, 16000),
}


def stft_mag(x: np.ndarray) -> np.ndarray:
    _, _, Z = signal.stft(x, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP,
                          boundary=None, padded=True)
    return np.abs(Z)


def detect_onsets(mono: np.ndarray, sensitivity: float = 1.0,
                  min_gap: float = 0.045) -> np.ndarray:
    """Multi-band spectral-flux onset detection. Returns onset sample indices."""
    S = stft_mag(np.concatenate([np.zeros(N_FFT), mono]))  # pad: catch hits at t=0
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    logS = np.log1p(100 * S)
    flux = np.maximum(0, np.diff(logS, axis=1))
    flux = np.pad(flux, ((0, 0), (1, 0)))

    # Sum flux per band and normalise each band separately so quiet hi-hats
    # are not drowned out by loud kicks.
    edges = [(20, 150), (150, 600), (600, 3000), (3000, 16000)]
    odf = np.zeros(S.shape[1])
    for lo, hi in edges:
        m = (freqs >= lo) & (freqs < hi)
        b = flux[m].sum(axis=0)
        b = b / (np.percentile(b, 99) + 1e-9)
        odf += b
    odf = signal.savgol_filter(odf, 5, 2) if len(odf) > 7 else odf
    odf = np.maximum(odf, 0)

    # Adaptive threshold: local median + offset
    win = max(3, int(0.25 * SR / HOP) | 1)
    med = signal.medfilt(odf, win)
    delta = 0.06 / max(sensitivity, 1e-3)
    thresh = med + delta * np.percentile(odf, 99.5)
    peaks, _ = signal.find_peaks(odf, height=thresh,
                                 distance=max(1, int(min_gap * SR / HOP)))
    # Convert frames to samples (frame i covers [i*HOP, i*HOP + N_FFT)).
    onsets = np.maximum(0, peaks * HOP + N_FFT // 2 - N_FFT)
    return reject_non_transients(mono, refine_onsets(mono, onsets))


def reject_non_transients(mono: np.ndarray, onsets: np.ndarray, ratio: float = 1.5) -> np.ndarray:
    """Drop onsets that are just the decay of the previous hit (double triggers).
    A real hit must clearly add energy in at least one frequency band, so it
    still passes when it sits on top of a loud bassline or pad."""
    bands = [("lp", 150), ("bp", (150, 1000)), ("bp", (1000, 5000)), ("hp", 5000)]
    filt = [signal.sosfilt(signal.butter(4, fc, kind, fs=SR, output="sos"), mono) for kind, fc in bands]
    w = int(0.012 * SR)
    gap = int(0.002 * SR)  # skip the pre-roll right before the onset
    keep = []
    for o in onsets:
        if o < w + gap:
            keep.append(o)
            continue
        best = 0.0
        for x in filt:
            post = np.sqrt(np.mean(x[o:o + w] ** 2))
            pre = np.sqrt(np.mean(x[o - w - gap:o - gap] ** 2)) + 1e-9
            best = max(best, post / pre)
        if best >= ratio:
            keep.append(o)
    return np.array(keep, dtype=int)


def _causal_rms(x: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    idx = np.arange(len(x)) + 1
    lo = np.maximum(0, idx - w)
    return np.sqrt((c[idx] - c[lo]) / np.maximum(1, idx - lo))


def refine_onsets(mono: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    """Move each onset to the start of its transient (~1 ms accuracy).

    Uses the rise of causal log-energy envelopes in four bands, so a loud
    bassline or pad underneath can't pull the onset onto its waveform."""
    if len(onsets) == 0:
        return onsets
    bands = [("lp", 150, 0.012), ("bp", (150, 1000), 0.006),
             ("bp", (1000, 5000), 0.003), ("hp", 5000, 0.002)]
    hop = int(0.0005 * SR)
    rise = None
    for kind, fc, win in bands:
        x = signal.sosfilt(signal.butter(4, fc, kind, fs=SR, output="sos"), mono)
        e = _causal_rms(x, int(win * SR))[::hop]
        le = np.log(e + 1e-3 * e.max() + 1e-12)
        d = np.maximum(0, np.diff(le, prepend=le[0]))
        rise = d if rise is None else rise + d
    k = 3
    rise = np.convolve(rise, np.ones(k) / k, mode="same")
    out = []
    back, fwd = int(0.030 * SR / hop), int(0.030 * SR / hop)
    for o in onsets:
        c = o // hop
        a = max(0, c - back)
        b = min(len(rise), c + fwd)
        if b <= a:
            continue
        seg = rise[a:b]
        j = int(np.argmax(seg))
        thr = seg[j] * 0.15
        lim = max(0, j - int(0.006 * SR / hop))
        while j > lim and seg[j - 1] > thr:
            j -= 1
        out.append(max(0, (a + j) * hop - int(0.001 * SR)))  # 1 ms pre-roll
    out = np.unique(np.array(out, dtype=int))
    keep = [out[0]] if len(out) else []
    for o in out[1:]:
        if o - keep[-1] > int(0.02 * SR):
            keep.append(o)
    return np.array(keep, dtype=int)


@dataclass
class Hit:
    index: int
    start: int                 # sample index
    gap_after: float           # seconds until next onset
    gap_before: float          # seconds since previous onset
    peak: float
    features: dict = field(default_factory=dict)
    label: str = "other"
    confidence: float = 0.0
    score: float = 0.0
    pitch_hz: float = 0.0


def psd(x: np.ndarray, nfft: int = 4096):
    """Power spectral density (comparable between segments of any length)."""
    n = min(len(x), nfft)
    f, p = signal.welch(x, fs=SR, nperseg=n, noverlap=n // 2, nfft=nfft)
    return f, p


def residual_psd(seg: np.ndarray, pre: np.ndarray | None, nfft: int = 4096):
    """PSD of the hit with the PSD of what was already sounding before it
    (tail of the previous hit, bass, pads, ...) subtracted."""
    f, p = psd(seg, nfft)
    if pre is not None and len(pre) >= 256:
        _, pp = psd(pre, nfft)
        p = np.maximum(p - 1.3 * pp, 1e-4 * p)
    return f, p


def band_energies(seg: np.ndarray, pre: np.ndarray | None = None) -> dict:
    """Band energy shares, centroid and flatness of the (residual) spectrum."""
    freqs, spec = residual_psd(seg, pre)
    total = spec[(freqs >= 20) & (freqs < 16000)].sum() + 1e-20
    out = {}
    for name, (lo, hi) in BANDS.items():
        e = float(spec[(freqs >= lo) & (freqs < hi)].sum())
        out[name] = e / total
        out["abs_" + name] = e
    out["centroid"] = float((freqs * spec).sum() / (spec.sum() + 1e-20))
    sp = spec[(freqs >= 100) & (freqs < 12000)] + 1e-20
    out["flatness"] = float(np.exp(np.mean(np.log(sp))) / np.mean(sp))
    return out


def decay_time(seg: np.ndarray, db: float = 30.0, floor: float = 0.0) -> float:
    """Seconds until the smoothed envelope falls `db` below its peak, or down to
    `floor` (e.g. the level of what was already ringing before the hit)."""
    env = np.abs(signal.hilbert(seg)) if len(seg) > 16 else np.abs(seg)
    k = max(1, int(0.005 * SR))
    env = np.convolve(env, np.ones(k) / k, mode="same")
    p = int(np.argmax(env))
    thr = max(env[p] * 10 ** (-db / 20), floor)
    below = np.where(env[p:] < thr)[0]
    return (below[0] if len(below) else len(env) - p) / SR


def fundamental(seg: np.ndarray, pre: np.ndarray | None = None,
                fmin=45.0, fmax=450.0) -> tuple[float, float]:
    """Dominant low-frequency peak of the residual spectrum (kick/tom tuning)
    and how tonal it is (peak / mean ratio)."""
    a = int(0.01 * SR)
    seg = seg[a:a + int(0.25 * SR)]
    if len(seg) < 1024:
        return 0.0, 0.0
    freqs, spec = residual_psd(seg, pre, nfft=16384)
    m = (freqs >= fmin) & (freqs <= fmax)
    if not m.any():
        return 0.0, 0.0
    amp = np.sqrt(spec[m])
    return float(freqs[m][np.argmax(amp)]), float(amp.max() / (amp.mean() + 1e-20))


def extract_features(mono: np.ndarray, hit: Hit):
    s = hit.start
    attack = mono[s:s + int(0.04 * SR)]            # first 40 ms: the "hit"
    body = mono[s:s + int(min(0.35, max(0.06, hit.gap_after)) * SR)]
    tail = mono[s + int(0.06 * SR): s + int(min(1.5, max(0.08, hit.gap_after)) * SR)]
    f = {}
    if len(attack) < 256:
        attack = np.pad(attack, (0, 256 - len(attack)))
    pre_len = int(min(0.06, max(0.0, hit.gap_before - 0.005)) * SR)
    pre = mono[s - pre_len:s] if pre_len >= 256 and s >= pre_len else None
    ea = band_energies(attack, pre)
    eb = band_energies(body if len(body) >= 256 else attack, pre)
    for k, v in ea.items():
        f["a_" + k] = v
    for k, v in eb.items():
        f["b_" + k] = v
    pre_w = mono[max(0, s - int(0.02 * SR)):s]
    pre_lvl = 1.6 * np.sqrt(np.mean(pre_w ** 2)) * np.sqrt(2) if len(pre_w) else 0.0
    f["decay"] = decay_time(mono[s:s + int(min(2.0, max(0.05, hit.gap_after)) * SR)], floor=pre_lvl)
    # Energy that keeps ringing after 60 ms, relative to the attack (open hat / cymbal)
    ra = np.sqrt(np.mean(attack ** 2)) + 1e-9
    rt = np.sqrt(np.mean(tail ** 2)) if len(tail) else 0.0
    f["sustain_ratio"] = float(rt / ra)
    # High-passed sustain (only the cymbal-ish part)
    if len(tail) > 512:
        sos = signal.butter(4, 5000, "hp", fs=SR, output="sos")
        th = signal.sosfilt(sos, tail)
        ah = signal.sosfilt(sos, attack)
        f["hf_sustain_ratio"] = float(np.sqrt(np.mean(th ** 2)) / (np.sqrt(np.mean(ah ** 2)) + 1e-9))
    else:
        f["hf_sustain_ratio"] = 0.0
    seg_full = mono[s:s + int(min(3.0, max(0.05, hit.gap_after)) * SR)]
    if len(seg_full) > 2048:
        sos = signal.butter(4, 4000, "hp", fs=SR, output="sos")
        a0 = max(0, s - int(0.03 * SR))
        hp = signal.sosfilt(sos, mono[a0:s + len(seg_full)])
        pre_hp = hp[: s - a0][-int(0.02 * SR):]
        floor = 1.6 * np.sqrt(2 * np.mean(pre_hp ** 2)) if len(pre_hp) else 0.0
        f["hf_decay"] = decay_time(hp[s - a0:], db=30, floor=floor)
    else:
        f["hf_decay"] = 0.0
    hit.pitch_hz, f["tonality"] = fundamental(body, pre)
    f["pitch"] = hit.pitch_hz
    hit.features = f


# ----------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------

def add_relative_features(hits: list[Hit]):
    """How big is this hit's *new* energy in each band compared with the
    strongest hits of the same track? (0..1+). Makes classification robust
    against bass notes / pads that start together with a hi-hat or snare."""
    for band in BANDS:
        vals = np.array([h.features["a_abs_" + band] for h in hits])
        ref = np.percentile(vals, 90) + 1e-20
        for h, v in zip(hits, vals):
            h.features["rel_" + band] = float(min(v / ref, 2.0))


def classify(hit: Hit):
    f = hit.features
    low = f["a_sub"] + f["a_low"]
    lowmid = f["a_lowmid"]
    mid = f["a_mid"]
    hi = f["a_high"] + f["a_air"]
    flat = f["a_flatness"]
    cen = f["a_centroid"]
    decay = f["decay"]
    hfs = f["hf_sustain_ratio"]
    pitch = f["pitch"]
    ton = f["tonality"]

    scores = {}
    # Kick: dominated by sub/low energy, low centroid, low pitch
    scores["kick"] = 2.2 * (f["b_sub"] + 0.6 * f["b_low"]) + 0.6 * low \
        - 1.2 * hi - 0.6 * mid + (0.4 if 0 < pitch < 95 else -0.3)
    # Snare: broadband noise plus body in 150-300 Hz, strong mid/high
    scores["snare"] = 1.4 * (mid + 0.7 * f["a_high"]) + 0.9 * lowmid + 1.2 * flat \
        - 1.2 * f["a_sub"] - 0.8 * f["a_air"] + (0.2 if 130 < pitch < 300 else 0) - 0.15 * ton / 10 \
        + 1.0 * min(low + lowmid, 0.3)
    # Hi-hats: almost all energy above 5 kHz, very little body
    hat_base = 2.2 * f["a_air"] - 0.8 * f["a_high"] - 4.0 * (low + lowmid) - 1.2 * mid
    hd = f.get("hf_decay", decay)
    scores["hihat_closed"] = hat_base + (0.5 if hd < 0.13 else -0.6 - 0.6 * min(hd, 1.5))
    scores["hihat_open"] = hat_base + (0.5 if 0.13 <= hd < 1.0 else -0.6)
    # Cymbals (crash/ride): bright, long, sustained high end, some mid
    scores["cymbal"] = 1.2 * hi + 0.6 * mid - 2.0 * low + 0.6 * min(hd, 2.0) \
        + 0.5 * min(hfs, 2.0) - (0.8 if hd < 0.6 else 0)
    # Toms: tonal low/low-mid, pitch between ~70 and 400 Hz, less noise than snare
    scores["tom"] = 1.6 * (f["b_low"] + f["b_lowmid"]) + 0.08 * min(ton, 20) \
        - 2.0 * hi - 0.8 * flat - 0.6 * f["b_sub"] + (0.4 if 75 < pitch < 400 else -0.6) \
        + (0.2 if decay > 0.15 else 0)

    # Track-relative evidence (only present when computed for the whole track)
    if "rel_sub" in f:
        rs, rl, rlm = f["rel_sub"], f["rel_low"], f["rel_lowmid"]
        rm, rh, ra = f["rel_mid"], f["rel_high"], f["rel_air"]
        scores["kick"] += 1.2 * (min(rs, 1) - 0.5)
        scores["tom"] += 0.8 * (min(max(rl, rlm), 1) - 0.5) - 1.5 * max(0, min(ra, 1) - 0.3)
        scores["snare"] += 0.8 * (min(max(rm, rh), 1) - 0.4)
        for k in ("hihat_closed", "hihat_open"):
            scores[k] += 1.0 * (min(ra, 1) - 0.4) - 0.8 * max(0, min(rs, 1) - 0.6)
        if rs < 0.25:
            scores["kick"] -= 1.0

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    hit.label = ranked[0][0]
    margin = ranked[0][1] - ranked[1][1]
    hit.confidence = float(1 / (1 + np.exp(-4 * margin)))  # 0.5 .. 1
    hit.features["class_scores"] = {k: round(v, 3) for k, v in scores.items()}


def split_toms(hits: list[Hit], max_toms: int = 4):
    """Cluster toms by pitch into tom_low / tom_mid / tom_high (etc.)."""
    toms = [h for h in hits if h.label == "tom" and h.pitch_hz > 0]
    if not toms:
        return
    pitches = np.log2(np.array([h.pitch_hz for h in toms]))
    # Pick the number of clusters with a simple gap heuristic (semitone based)
    order = np.sort(pitches)
    gaps = np.diff(order) * 12  # semitones
    big_gaps = np.where(gaps > 2.5)[0]
    k = min(max_toms, len(big_gaps) + 1, len(toms))
    if k <= 1:
        for h in toms:
            h.label = "tom"
        return
    cut_idx = np.sort(big_gaps[np.argsort(gaps[big_gaps])[::-1][:k - 1]])
    boundaries = [(order[i] + order[i + 1]) / 2 for i in cut_idx]
    names = {2: ["tom_low", "tom_high"],
             3: ["tom_low", "tom_mid", "tom_high"],
             4: ["tom_floor", "tom_low", "tom_mid", "tom_high"]}[k]
    for h, p in zip(toms, pitches):
        h.label = names[int(np.searchsorted(boundaries, p))]


def layered_highs(hits: list[Hit]) -> list[Hit]:
    """Hi-hats and cymbals are often played together with a kick/snare/tom (or
    on a bass note) and the hit gets the drum's label. If a strong high layer
    is present, add a secondary hi-hat / cymbal candidate. The per-element
    high-pass filter on export removes most of the drum underneath."""
    extra = []
    for h in hits:
        if h.label in ("cymbal", "hihat_open", "hihat_closed"):
            continue
        f = h.features
        hd = f.get("hf_decay", 0)
        label = None
        if hd > 0.6 and f["b_high"] + f["b_air"] > 0.04 and h.gap_after > 0.6:
            label = "cymbal"
        elif f.get("rel_air", 0) > 0.35 and f["a_air"] > 0.08:
            label = "hihat_open" if hd >= 0.13 else "hihat_closed"
        if label:
            c = Hit(h.index, h.start, h.gap_after, h.gap_before, h.peak, dict(f), label,
                    h.confidence * 0.6, 0.0, h.pitch_hz)
            c.features["layered_with"] = h.label
            extra.append(c)
    return extra


def split_cymbals(hits: list[Hit]):
    """Rides ring with a distinct bell/ping and repeat fast; crashes are long washes."""
    for h in hits:
        if h.label != "cymbal":
            continue
        if h.features["decay"] > 0.9 and h.features["a_flatness"] > 0.25:
            h.label = "crash"
        elif h.gap_after < 0.6:
            h.label = "ride"
        else:
            h.label = "crash"


# ----------------------------------------------------------------------------
# Sample cutting & ranking
# ----------------------------------------------------------------------------

MAX_LEN = {  # max sample length per element (seconds)
    "kick": 1.0, "snare": 1.0, "hihat_closed": 0.4, "hihat_open": 1.5,
    "crash": 4.0, "ride": 2.5, "cymbal": 3.0, "tom": 1.5, "tom_low": 1.8,
    "tom_mid": 1.5, "tom_high": 1.2, "tom_floor": 2.0, "other": 1.0,
}

CLEAN_FILTERS = {  # optional bleed-reduction filters per element
    "hihat_closed": ("hp", 2500), "hihat_open": ("hp", 2500),
    "crash": ("hp", 400), "ride": ("hp", 400), "cymbal": ("hp", 400),
    "snare": ("hp", 90), "tom_high": ("hp", 60), "tom_mid": ("hp", 50),
}


def cut_sample(audio: np.ndarray, hit: Hit, clean: bool) -> np.ndarray:
    max_len = MAX_LEN.get(hit.label, 1.0)
    length = int(min(max_len, max(hit.gap_after, 0.05)) * SR)
    seg = audio[:, hit.start:hit.start + length].copy()
    # Trim trailing silence (-60 dB)
    env = np.max(np.abs(seg), axis=0)
    if env.max() > 0:
        above = np.where(env > env.max() * 1e-3)[0]
        if len(above):
            seg = seg[:, :above[-1] + int(0.01 * SR)]
    if clean and hit.label in CLEAN_FILTERS:
        kind, fc = CLEAN_FILTERS[hit.label]
        sos = signal.butter(2, fc, kind, fs=SR, output="sos")
        seg = signal.sosfiltfilt(sos, seg, axis=1) if seg.shape[1] > 30 else seg
    # Fades: 0.5 ms in, up to 15 ms out
    n = seg.shape[1]
    fi = min(n, int(0.0005 * SR))
    fo = min(n, int(0.015 * SR))
    if fi:
        seg[:, :fi] *= np.linspace(0, 1, fi)
    if fo:
        seg[:, -fo:] *= np.linspace(1, 0, fo) ** 2
    return seg


def rank_hits(hits: list[Hit]):
    """Higher score = cleaner, more representative sample."""
    if not hits:
        return
    by_label: dict[str, list[Hit]] = {}
    for h in hits:
        by_label.setdefault(h.label, []).append(h)
    for label, group in by_label.items():
        want = min(MAX_LEN.get(label, 1.0), 0.5)
        peaks = np.array([h.peak for h in group])
        ref = np.percentile(peaks, 90) + 1e-9
        # Typical feature vector of the class (median) -> prefer typical hits
        keys = ["a_sub", "a_low", "a_lowmid", "a_mid", "a_high", "a_air"]
        mat = np.array([[h.features[k] for k in keys] for h in group])
        med = np.median(mat, axis=0)
        for h, row in zip(group, mat):
            isolation = min(1.0, h.gap_after / want) * (0.7 + 0.3 * min(1.0, h.gap_before / 0.1))
            loud = min(1.0, h.peak / (0.6 * ref))   # quiet = ghost note / bleed
            typical = float(np.exp(-4 * np.abs(row - med).sum()))
            layered = 0.5 if "layered_with" in h.features else 1.0
            h.score = float((0.45 * isolation + 0.3 * h.confidence + 0.25 * typical)
                            * loud * layered)


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

def extract_drumkit(input_path: str | Path, out_dir: str | Path | None = None,
                    per_class: int = 3, separate: str = "auto",
                    sensitivity: float = 1.0, clean: bool = True,
                    bits: int = 24, export_all: bool = False,
                    hpss_mode: str = "auto", log=print) -> dict:
    input_path = Path(input_path)
    out_dir = Path(out_dir) if out_dir else input_path.with_name(input_path.stem + "_drumkit")
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading {input_path.name} ...")
    audio = load_audio(input_path)
    log(f"  {audio.shape[1] / SR:.1f} s, stereo, {SR} Hz")

    source = "mix"
    if separate in ("auto", "yes"):
        drums = separate_drums(input_path, log=log)
        if drums is not None:
            audio, source = drums, "demucs drum stem"
            write_wav(out_dir / "_drum_stem.wav", audio, bits=bits)
        elif separate == "yes":
            raise RuntimeError("Demucs not installed (pip install demucs).")
        else:
            log("  Demucs not installed - analysing the full mix. For full songs, "
                "`pip install demucs` gives much cleaner samples.")

    if source == "mix" and hpss_mode != "no":
        log("Separating percussive part (HPSS) ...")
        perc, mixness = hpss(audio)
        log(f"  non-drum floor between hits: {mixness:.2f} (>0.28 = full mix)")
        if hpss_mode == "yes" or mixness > 0.28:
            audio, source = perc, "HPSS percussive part"
            log("  -> full mix detected, using percussive part (use --no-hpss for drum-only files)")
        else:
            log("  -> looks like a drum-only recording, keeping original audio")

    mono = audio.mean(axis=0)
    norm = np.max(np.abs(mono)) + 1e-9
    mono_n = mono / norm

    log("Detecting hits ...")
    onsets = detect_onsets(mono_n, sensitivity=sensitivity)
    log(f"  {len(onsets)} hits found")
    if len(onsets) == 0:
        return {"hits": 0}

    hits: list[Hit] = []
    for i, s in enumerate(onsets):
        nxt = onsets[i + 1] if i + 1 < len(onsets) else len(mono_n)
        prv = onsets[i - 1] if i > 0 else -SR
        pk = float(np.max(np.abs(mono_n[s:s + int(0.03 * SR)]))) if s < len(mono_n) else 0.0
        hits.append(Hit(i, int(s), (nxt - s) / SR, (s - prv) / SR, pk))

    log("Analysing & classifying ...")
    for h in hits:
        extract_features(mono_n, h)
    add_relative_features(hits)
    for h in hits:
        classify(h)
    # Drop very quiet hits (ghost notes / noise) from export candidates
    peaks = np.array([h.peak for h in hits])
    floor = 0.05 * np.percentile(peaks, 95)
    hits = [h for h in hits if h.peak >= floor]
    split_toms(hits)
    hits += layered_highs(hits)
    hits.sort(key=lambda h: h.start)
    split_cymbals(hits)
    rank_hits(hits)

    # Export
    summary = {}
    labels = sorted(set(h.label for h in hits))
    for label in labels:
        group = sorted([h for h in hits if h.label == label], key=lambda h: h.score, reverse=True)
        chosen = group if export_all else group[:per_class]
        d = out_dir / label
        d.mkdir(exist_ok=True)
        for rank, h in enumerate(chosen, 1):
            seg = cut_sample(audio, h, clean)
            peak = np.max(np.abs(seg)) + 1e-9
            seg = seg / peak * 0.891  # normalise to -1 dBFS
            name = f"{label}_{rank:02d}_t{h.start / SR:07.3f}s.wav"
            write_wav(d / name, seg, bits=bits)
        summary[label] = {"detected": len(group), "exported": len(chosen)}
        log(f"  {label:13s} {len(group):4d} hits -> {len(chosen)} samples")

    report = {
        "input": str(input_path), "source_analysed": source,
        "duration_s": round(audio.shape[1] / SR, 2), "total_hits": len(hits),
        "elements": summary,
        "hits": [{"time_s": round(h.start / SR, 4), "label": h.label,
                  "confidence": round(h.confidence, 3), "score": round(h.score, 3),
                  "pitch_hz": round(h.pitch_hz, 1), "decay_s": round(h.features["decay"], 3)}
                 for h in hits],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    try:
        plot_overview(mono_n, hits, out_dir / "overview.png")
    except Exception as e:  # matplotlib optional
        log(f"  (overview image skipped: {e})")
    log(f"Done -> {out_dir}")
    return report


COLORS = {"kick": "#d62728", "snare": "#1f77b4", "hihat_closed": "#2ca02c",
          "hihat_open": "#98df8a", "crash": "#ff7f0e", "ride": "#ffbb78",
          "cymbal": "#ff7f0e", "tom": "#9467bd", "tom_low": "#6a3d9a",
          "tom_mid": "#9467bd", "tom_high": "#c5b0d5", "tom_floor": "#3f007d",
          "other": "#7f7f7f"}


def plot_overview(mono, hits, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(mono)) / SR
    step = max(1, len(mono) // 20000)
    labels = sorted(set(h.label for h in hits))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.4]})
    ax1.plot(t[::step], mono[::step], color="#999", lw=0.5)
    for h in hits:
        ax1.axvline(h.start / SR, color=COLORS.get(h.label, "k"), lw=0.8, alpha=0.8)
    ax1.set_ylabel("waveform")
    ax1.set_title("Detected drum hits")
    for h in hits:
        y = labels.index(h.label)
        ax2.scatter(h.start / SR, y, color=COLORS.get(h.label, "k"), s=12 + 30 * h.score)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("time (s)")
    ax2.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Minimal desktop GUI (tkinter)
# ----------------------------------------------------------------------------

def run_gui():
    import threading
    import tkinter as tk
    from tkinter import filedialog, ttk, scrolledtext

    root = tk.Tk()
    root.title("Drumkit Extractor")
    root.geometry("640x480")
    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)

    in_var, out_var = tk.StringVar(), tk.StringVar()
    n_var = tk.IntVar(value=3)
    sens_var = tk.DoubleVar(value=1.0)
    clean_var = tk.BooleanVar(value=True)
    sep_var = tk.BooleanVar(value=demucs_available())

    def pick_in():
        p = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.flac *.aif *.aiff *.m4a *.ogg"), ("All", "*.*")])
        if p:
            in_var.set(p)
            out_var.set(str(Path(p).with_name(Path(p).stem + "_drumkit")))

    def pick_out():
        p = filedialog.askdirectory()
        if p:
            out_var.set(p)

    ttk.Label(frm, text="Audio file").grid(row=0, column=0, sticky="w")
    ttk.Entry(frm, textvariable=in_var, width=55).grid(row=0, column=1, sticky="ew")
    ttk.Button(frm, text="Browse...", command=pick_in).grid(row=0, column=2)
    ttk.Label(frm, text="Output folder").grid(row=1, column=0, sticky="w")
    ttk.Entry(frm, textvariable=out_var, width=55).grid(row=1, column=1, sticky="ew")
    ttk.Button(frm, text="Browse...", command=pick_out).grid(row=1, column=2)
    ttk.Label(frm, text="Samples per element").grid(row=2, column=0, sticky="w")
    ttk.Spinbox(frm, from_=1, to=50, textvariable=n_var, width=6).grid(row=2, column=1, sticky="w")
    ttk.Label(frm, text="Sensitivity").grid(row=3, column=0, sticky="w")
    ttk.Scale(frm, from_=0.3, to=3.0, variable=sens_var).grid(row=3, column=1, sticky="ew")
    ttk.Checkbutton(frm, text="Reduce bleed (filters per element)", variable=clean_var).grid(row=4, column=1, sticky="w")
    sep_text = "Separate drums with Demucs" + ("" if demucs_available() else " (not installed)")
    cb = ttk.Checkbutton(frm, text=sep_text, variable=sep_var)
    cb.grid(row=5, column=1, sticky="w")
    if not demucs_available():
        cb.state(["disabled"])

    logbox = scrolledtext.ScrolledText(frm, height=14)
    logbox.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
    frm.columnconfigure(1, weight=1)
    frm.rowconfigure(7, weight=1)

    def log(msg):
        root.after(0, lambda: (logbox.insert("end", msg + "\n"), logbox.see("end")))

    def go():
        if not in_var.get():
            log("Pick an audio file first.")
            return
        btn.state(["disabled"])

        def work():
            try:
                extract_drumkit(in_var.get(), out_var.get() or None, per_class=n_var.get(),
                                separate="auto" if sep_var.get() else "no",
                                sensitivity=sens_var.get(), clean=clean_var.get(), log=log)
                out = out_var.get()
                if out and sys.platform == "darwin":
                    subprocess.run(["open", out])
                elif out and os.name == "nt":
                    os.startfile(out)  # type: ignore[attr-defined]
            except Exception as e:
                log(f"ERROR: {e}")
            finally:
                root.after(0, lambda: btn.state(["!disabled"]))

        threading.Thread(target=work, daemon=True).start()

    btn = ttk.Button(frm, text="Extract drumkit", command=go)
    btn.grid(row=6, column=1, sticky="w", pady=(8, 0))
    root.mainloop()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] == "gui":
        try:
            return run_gui()
        except ImportError:
            print("tkinter is not available; use the command line (see --help).")
            if not argv:
                return 1
    ap = argparse.ArgumentParser(description="Extract a drum kit (one-shot samples) from an audio file.")
    ap.add_argument("input", help="audio file (wav, mp3, flac, m4a, ...)")
    ap.add_argument("-o", "--out", help="output folder (default: <input>_drumkit)")
    ap.add_argument("-n", "--per-class", type=int, default=3, help="samples to export per element (default 3)")
    ap.add_argument("--all", action="store_true", help="export every detected hit")
    ap.add_argument("-s", "--sensitivity", type=float, default=1.0, help="onset sensitivity, >1 finds more hits")
    sep = ap.add_mutually_exclusive_group()
    sep.add_argument("--separate", action="store_true", help="require Demucs drum separation")
    sep.add_argument("--no-separate", action="store_true", help="skip drum separation")
    ap.add_argument("--raw", action="store_true", help="don't apply bleed-reduction filters")
    hp = ap.add_mutually_exclusive_group()
    hp.add_argument("--hpss", action="store_true", help="always strip bass/pads/vocals with HPSS (when not using Demucs)")
    hp.add_argument("--no-hpss", action="store_true", help="never use HPSS (drum-only recordings)")
    ap.add_argument("--bits", type=int, choices=[16, 24], default=24)
    a = ap.parse_args(argv)
    extract_drumkit(a.input, a.out, per_class=a.per_class,
                    separate="yes" if a.separate else ("no" if a.no_separate else "auto"),
                    sensitivity=a.sensitivity, clean=not a.raw, bits=a.bits, export_all=a.all,
                    hpss_mode="yes" if a.hpss else ("no" if a.no_hpss else "auto"))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # needed for packaged builds (torch/demucs)
    sys.exit(main())
