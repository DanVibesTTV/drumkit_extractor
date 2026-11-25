# Drumkit Extractor {{VERSION}}

Turn any song or drum recording into a ready-to-use drum kit. Drumkit Extractor finds every drum hit in an audio file, sorts the hits into **kick, snare, closed/open hi-hat, toms (floor/low/mid/high), crash and ride**, and exports the cleanest examples of each as WAV one-shots.

## Downloads

| File | What it is |
|---|---|
| **`DrumkitExtractor.exe`** | Double-click app with a window. No installation needed. **Start here.** |
| `drumkit-cli.exe` | Same engine for the command line and batch processing (`drumkit-cli.exe song.mp3 -n 5`) |
| `DrumkitExtractor-full-win64.zip` | Large download that includes **Demucs** AI drum separation, for the cleanest results on full songs. Unzip it and run `DrumkitExtractor.exe` inside the folder. |

Requirements: Windows 10 or 11, 64-bit. Everything else is built in.

> **SmartScreen warning?** The app isn't code-signed, so Windows may say it "protected your PC" the first time. Click **More info → Run anyway**.

## Features

- Reads WAV, MP3, FLAC, M4A, OGG, AIFF and more
- Automatic drum isolation: built-in bass/pad removal for full mixes, or Demucs in the full version. Drum-only recordings are detected and left untouched.
- Classifies hits into kick, snare, hi-hats, toms (sorted by pitch), crash and ride
- Picks the cleanest, most isolated hits and exports them as trimmed, faded, normalised 24-bit WAV files
- `report.json` with every detected hit, plus an `overview.png` timeline
- Options for samples per element, sensitivity, bleed-reduction filters and 16/24-bit output

## Tips

- Best results come from drum-only tracks, drum breaks and fills, or the full version on complete songs.
- Hits played at the same moment (e.g. kick + crash) can't be fully separated. The app picks the most isolated examples it can find.
- Not happy with a sample? Export more candidates (`-n 10`) and pick your favourite.

---

## Third-party software & licences

Drumkit Extractor bundles **FFmpeg** to decode audio files. FFmpeg is included as an unmodified, separate program (`ffmpeg.exe`) that Drumkit Extractor runs as a subprocess.

- **Version:** FFmpeg {{FFMPEG_VERSION}} (Windows "essentials" build by [gyan.dev](https://www.gyan.dev/ffmpeg/builds/))
- **Licence:** GNU General Public License v3 (GPLv3). The full licence text is attached to this release as `ffmpeg-LICENSE.txt`.
- **Source code:** the FFmpeg source for this exact version is attached to this release as `{{FFMPEG_SOURCE_FILE}}`, and is also available at <https://ffmpeg.org/releases/>. The build configuration and the versions of the external libraries in this build are documented at <https://www.gyan.dev/ffmpeg/builds/>.
- FFmpeg is a trademark of Fabrice Bellard, originator of the FFmpeg project. More at <https://ffmpeg.org>.

You may redistribute FFmpeg under the terms of the GPLv3. The source-code offer above applies to the FFmpeg binary included in this release.

The **full** version also bundles [Demucs](https://github.com/facebookresearch/demucs) (MIT License) and [PyTorch](https://pytorch.org) (BSD-style licence). Both versions include Python (PSF License), NumPy and SciPy (BSD) and Matplotlib (Matplotlib License, BSD-compatible).
