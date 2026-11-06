"""Clonebox: voice cloning web app on Chatterbox Turbo.

    uv run uvicorn app:app --host 127.0.0.1 --port 8000
"""

import asyncio
import json
import os
import random
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import soundfile as sf
import structlog
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals
from fastapi import FastAPI, File, Form, HTTPException, Path as PathParam, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from core import TAG_RE, clean_reference, normalize_tags

log = structlog.get_logger()

DATA = Path(os.environ.get("CLONEBOX_DATA", Path(__file__).parent / "data"))
VOICES = DATA / "voices"
CLIPS = DATA / "clips"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_LINE_CHARS = 600
MAX_JOIN_CLIPS = 300
FFMPEG_TIMEOUT_S = 60
ID_PATTERN = r"^[0-9a-f]{32}$"
IdParam = Annotated[str, PathParam(pattern=ID_PATTERN)]


class State:
    model: ChatterboxTurboTTS
    known_tags: list[str]
    lock = asyncio.Lock()  # one model, one device: serialize all inference
    conds: dict[str, Conditionals] = {}


state = State()


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@asynccontextmanager
async def lifespan(_: FastAPI):
    VOICES.mkdir(parents=True, exist_ok=True)
    CLIPS.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("model.loading", device=device)
    state.model = await asyncio.to_thread(ChatterboxTurboTTS.from_pretrained, device=device)
    state.known_tags = sorted(t for t in state.model.tokenizer.get_added_vocab() if TAG_RE.fullmatch(t))
    log.info("model.ready", tags=len(state.known_tags))
    yield


app = FastAPI(title="Clonebox", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def problem_json(_, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        {"title": exc.detail, "status": exc.status_code},
        status_code=exc.status_code,
        media_type="application/problem+json",
    )


# ---------- voices ----------


def voice_dir(voice_id: str) -> Path:
    d = VOICES / voice_id
    if not d.is_dir():
        raise HTTPException(404, "voice not found")
    return d


def load_conds(voice_id: str) -> Conditionals:
    if voice_id not in state.conds:
        path = voice_dir(voice_id) / "conds.pt"
        state.conds[voice_id] = Conditionals.load(path, map_location=state.model.device).to(state.model.device)
    return state.conds[voice_id]


@app.get("/api/voices")
async def list_voices() -> list[dict]:
    metas = [json.loads(p.read_text()) for p in VOICES.glob("*/meta.json")]
    return sorted(metas, key=lambda m: m["created"], reverse=True)


async def decode_to_wav(src: Path, dst: Path) -> None:
    """Browser recordings arrive as webm/mp4; ffmpeg normalizes everything to mono wav."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "24000", str(dst),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, err = await asyncio.wait_for(proc.communicate(), FFMPEG_TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        raise HTTPException(422, "audio decode timed out")
    if proc.returncode != 0:
        log.warning("voice.decode_failed", stderr=err.decode(errors="replace")[-300:])
        raise HTTPException(422, "could not decode audio file")


@app.post("/api/voices", status_code=201)
async def create_voice(
    name: Annotated[str, Form(min_length=1, max_length=60)],
    consent: Annotated[bool, Form()],
    file: Annotated[UploadFile, File()],
) -> dict:
    if not consent:
        raise HTTPException(422, "confirm you have the speaker's consent to clone this voice")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "audio file too large (max 25 MB)")

    voice_id = uuid.uuid4().hex
    d = VOICES / voice_id
    with tempfile.TemporaryDirectory() as tmp:
        src, wav = Path(tmp) / "upload", Path(tmp) / "decoded.wav"
        src.write_bytes(raw)
        await decode_to_wav(src, wav)
        d.mkdir(parents=True)
        try:
            duration = await asyncio.to_thread(clean_reference, wav, d / "reference.wav")
            async with state.lock:
                conds = await asyncio.to_thread(_prepare, d)
        except ValueError as e:
            shutil.rmtree(d)
            raise HTTPException(422, str(e))
        except BaseException:
            shutil.rmtree(d, ignore_errors=True)
            raise

    state.conds[voice_id] = conds
    meta = {"id": voice_id, "name": name.strip(), "duration": round(duration, 1), "created": time.time()}
    (d / "meta.json").write_text(json.dumps(meta))
    log.info("voice.created", voice_id=voice_id, duration=meta["duration"])
    return meta


def _prepare(d: Path) -> Conditionals:
    state.model.prepare_conditionals(str(d / "reference.wav"))
    state.model.conds.save(d / "conds.pt")
    return state.model.conds


@app.get("/api/voices/{voice_id}/reference")
async def voice_reference(voice_id: IdParam) -> FileResponse:
    return FileResponse(voice_dir(voice_id) / "reference.wav", media_type="audio/wav")


@app.delete("/api/voices/{voice_id}", status_code=204)
async def delete_voice(voice_id: IdParam) -> None:
    shutil.rmtree(voice_dir(voice_id))
    state.conds.pop(voice_id, None)
    log.info("voice.deleted", voice_id=voice_id)


# ---------- synthesis ----------


@app.get("/api/tags")
async def tags() -> list[str]:
    return state.known_tags


class TTSRequest(BaseModel):
    voice_id: str = Field(pattern=ID_PATTERN)
    text: str = Field(min_length=1, max_length=MAX_LINE_CHARS)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)


def _synth(conds: Conditionals, text: str, seed: int, temperature: float) -> np.ndarray:
    torch.manual_seed(seed)
    state.model.conds = conds
    return state.model.generate(text, temperature=temperature).squeeze(0).numpy()


@app.post("/api/tts")
async def tts(req: TTSRequest) -> dict:
    text, dropped = normalize_tags(req.text, set(state.known_tags))
    if not text or not TAG_RE.sub("", text).strip():
        raise HTTPException(422, "nothing to say after removing unsupported tags")
    seed = req.seed if req.seed is not None else random.randrange(2**31)

    async with state.lock:
        conds = await asyncio.to_thread(load_conds, req.voice_id)
        wav = await asyncio.to_thread(_synth, conds, text, seed, req.temperature)

    clip_id = uuid.uuid4().hex
    sf.write(CLIPS / f"{clip_id}.wav", wav, state.model.sr)
    duration = round(len(wav) / state.model.sr, 2)
    log.info("tts.done", voice_id=req.voice_id, chars=len(text), seconds=duration)
    return {"clip_id": clip_id, "seed": seed, "duration": duration, "text": text, "dropped_tags": dropped}


class JoinRequest(BaseModel):
    clip_ids: list[Annotated[str, Field(pattern=ID_PATTERN)]] = Field(min_length=1, max_length=MAX_JOIN_CLIPS)
    gap_s: float = Field(default=0.4, ge=0, le=5)
    # Per-clip pause *after* each clip; overrides gap_s. Same length as clip_ids (last entry unused).
    gaps_s: list[Annotated[float, Field(ge=0, le=5)]] | None = None

    @model_validator(mode="after")
    def _gaps_match(self) -> "JoinRequest":
        if self.gaps_s is not None and len(self.gaps_s) != len(self.clip_ids):
            raise ValueError("gaps_s must have one entry per clip")
        return self


def _join(paths: list[Path], gaps_s: list[float], dst: Path) -> float:
    sr = state.model.sr
    parts = []
    for p, g in zip(paths, gaps_s):
        wav, _ = sf.read(p, dtype="float32")
        parts += [wav, np.zeros(int(g * sr), dtype=np.float32)]
    out = np.concatenate(parts[:-1])
    sf.write(dst, out, sr)
    return len(out) / sr


@app.post("/api/join")
async def join(req: JoinRequest) -> dict:
    paths = [CLIPS / f"{c}.wav" for c in req.clip_ids]
    if missing := [p.stem for p in paths if not p.is_file()]:
        raise HTTPException(404, f"clips not found: {', '.join(missing[:5])}")
    clip_id = uuid.uuid4().hex
    duration = await asyncio.to_thread(_join, paths, req.gaps_s or [req.gap_s] * len(paths), CLIPS / f"{clip_id}.wav")
    return {"clip_id": clip_id, "duration": round(duration, 2)}


@app.get("/api/clips/{clip_id}")
async def clip(clip_id: IdParam) -> FileResponse:
    path = CLIPS / f"{clip_id}.wav"
    if not path.is_file():
        raise HTTPException(404, "clip not found")
    return FileResponse(path, media_type="audio/wav", filename=f"clonebox_{clip_id[:8]}.wav")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
