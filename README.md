# Clonebox

Clone a voice from a short recording and use it to read multi-line scripts. Clonebox runs locally as a FastAPI app with a single-page UI on top of [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox).

- **Voice cloning from 6–15 s of speech.** You can record in the browser or upload a file. Clonebox removes long silences, keeps up to 15 s of speech and normalizes the loudness to −20 LUFS.
- **Script editor that generates each line separately.** You can regenerate any line with a new seed without touching the other lines.
- **Expression tags** such as `[laugh]`, `[sigh]` and `[dramatic]`. Clonebox maps unknown tags to the closest supported tag or removes them, so the model never reads a tag aloud.
- **Pause control.** End a line with `{0.8}` to put a 0.8 s pause after it. You can also set a default gap for all lines.
- **Join and export.** Joins all the lines into one WAV file you can download.
- **Runs entirely on your machine.** It uses CUDA, Apple Silicon (MPS) or the CPU, and audio never leaves your machine.

## Quickstart

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and `ffmpeg` on your `PATH`.

```bash
git clone https://github.com/logeshkannan96/clonebox.git
cd clonebox
uv run app.py
```

`uv run` installs dependencies on the first run. Open http://127.0.0.1:8000. The first start downloads the Chatterbox Turbo weights from Hugging Face, which takes a few minutes.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `CLONEBOX_DATA` | `./data` | Where voices (`voices/<id>/`) and generated clips (`clips/`) are stored |
| `CLONEBOX_HOST` | `127.0.0.1` | Bind address (see [SECURITY.md](SECURITY.md) before changing) |
| `CLONEBOX_PORT` | `8000` | Port |

## API

The UI is a thin client over these endpoints. Errors are returned as `application/problem+json`. Interactive docs are at `/docs`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/voices` | List voices |
| `POST` | `/api/voices` | Create a voice (multipart: `name`, `consent=true`, `file`, max 25 MB) |
| `GET` | `/api/voices/{id}/reference` | Cleaned reference audio |
| `DELETE` | `/api/voices/{id}` | Delete a voice |
| `GET` | `/api/tags` | Expression tags the model supports |
| `POST` | `/api/tts` | `{voice_id, text, seed?, temperature?}` → `{clip_id, seed, duration, text, dropped_tags}` |
| `POST` | `/api/join` | `{clip_ids, gap_s?, gaps_s?}` → `{clip_id, duration}` |
| `GET` | `/api/clips/{id}` | Download a clip as WAV |

## How it works

```
upload ─▶ ffmpeg (mono 24 kHz) ─▶ clean_reference (trim silence, ≤15 s, −20 LUFS) ─▶ prepare_conditionals ─▶ conds.pt
script line ─▶ normalize_tags ─▶ Chatterbox Turbo (cached conds, seeded) ─▶ clip.wav ─▶ join
```

The app loads one model instance at startup. An `asyncio.Lock` makes inference run one request at a time, because the model holds per-voice conditionals as mutable state. As a result, throughput is one generation at a time per process.

## Limitations

- **No authentication.** It's built for `localhost`. See [SECURITY.md](SECURITY.md) before exposing it on a network.
- **Generated clips are never deleted automatically.** Clean up `data/clips/` yourself.
- **Single process and a single model.** Don't run it with multiple uvicorn workers, because each worker loads its own copy of the model.

## Responsible use

Clone only your own voice or the voice of someone who has given explicit consent. The app requires you to confirm consent before it creates a voice. Chatterbox embeds an imperceptible [Perth](https://github.com/resemble-ai/perth) watermark in the generated audio. Don't use Clonebox for impersonation, fraud or deception.

## License

[MIT](LICENSE). Bundled third-party assets keep their own licenses: see `static/vendor/*-LICENSE.txt`.
