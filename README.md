# Drumkit Extractor

Turns an audio file into a folder of drum one-shots: kick, snare, closed and
open hi-hat, toms (sorted by pitch into floor / low / mid / high), crash and ride.

## Install

1. Python 3.9+ and **ffmpeg** on your PATH
   (macOS: `brew install ffmpeg`, Windows: `winget install ffmpeg`, Linux: your package manager)
2. `pip install -r requirements.txt`
3. Recommended for full songs: `pip install demucs`. This installs PyTorch, roughly 2 GB.
   Demucs separates the drum stem before analysis, which is the biggest single
   quality boost for mixed music.

## Build a standalone Windows .exe

The finished program needs nothing installed: Python, numpy/scipy and ffmpeg are all inside it.

**Option A: on your own Windows PC.** Install Python 3.12 from python.org, then double-click
`build_windows.bat` (or run it in a terminal). After a few minutes you get:

- `dist\DrumkitExtractor.exe`: double-click app with a window (~120 MB, mostly ffmpeg)
- `dist\drumkit-cli.exe`: the same thing for the command line

`build_windows.bat full` also bundles Demucs, PyTorch and the model, for the best
results on full songs. Because it's about 2 GB, it is built as a folder:
`dist\DrumkitExtractor-full\DrumkitExtractor.exe`. Zip that folder to move it to another PC.

**Option B: no Python at all.** Push this folder to a GitHub repository, open
**Actions → Build Windows exe → Run workflow**, and download the .exe from the
finished run's *Artifacts*. Tick "Include Demucs" for the full build.

Notes: Windows SmartScreen may warn about an unsigned .exe the first time
("More info → Run anyway"). The bundled ffmpeg is a GPL build. If you share the .exe
publicly, include a link to ffmpeg's source (ffmpeg.org).

## Use

```bash
python drumkit_extractor.py song.mp3              # -> song_drumkit/
python drumkit_extractor.py song.mp3 -n 5         # 5 samples per element
python drumkit_extractor.py drums.wav --no-hpss   # drum-only recording, keep it untouched
python drumkit_extractor.py song.wav --all        # every detected hit
python drumkit_extractor.py gui                   # simple desktop window
```

| Option | What it does |
|---|---|
| `-n N` | samples exported per element (default 3, best first) |
| `-s 1.5` | higher sensitivity finds quieter hits (ghost notes); lower is stricter |
| `--separate` / `--no-separate` | force / skip Demucs |
| `--hpss` / `--no-hpss` | force / skip the built-in bass+pad removal (auto-detected otherwise) |
| `--raw` | no bleed-reduction filters (e.g. the high-pass on hi-hats) |
| `--bits 16` | 16-bit instead of 24-bit WAV |

## Output

```
song_drumkit/
  kick/kick_01_t012.345s.wav   <- rank 01 = cleanest, t = position in the song
  snare/ ...  hihat_closed/ ...  hihat_open/ ...  tom_low/ ...  crash/ ...
  report.json                  <- every hit: time, label, confidence, pitch, decay
  overview.png                 <- timeline of all classified hits
  _drum_stem.wav               <- only when Demucs was used
```

Samples are trimmed, faded and normalised to -1 dBFS.

## How it works

1. **Separation:** Demucs if available, otherwise median-filter HPSS.
2. **Onset detection:** multi-band spectral flux, then a band-wise "is this a new
   transient?" check (removes double triggers) and ~1 ms onset refinement.
3. **Features per hit:** band energies with the previous hit's tail subtracted,
   centroid, flatness, decay, high-frequency decay, pitch and tonality, plus how
   strong each band is compared with the rest of the track.
4. **Classification:** a rule-based score per element. Toms are clustered by pitch.
   Hits that layer a cymbal or hat over a drum also become secondary candidates.
5. **Ranking:** isolation from neighbouring hits, confidence, typicality and loudness.

## Tips & limits

- Best results: drum-only recordings, drum breaks, or songs run through Demucs.
- Hits played together (kick + crash) can't be fully separated. The tool picks the
  most isolated examples, so longer songs with breaks or fills give better kits.
- Samples are cut where the next hit starts. Long tails come out best from sparse passages.
- If something is misfiled, `-n 10` gives you more candidates to choose from.
  `report.json` shows the per-class scores of each hit.
