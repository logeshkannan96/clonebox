"""Model-free audio/text helpers, lifted from Qwen3-TTS/chatterbox_clone.py."""

import re
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

REF_SR = 24_000
SILENCE_TOP_DB = 40  # quieter than peak by this much = silence
KEEP_GAP_S = 0.4  # pauses shorter than this stay in (natural phrasing)
MAX_REF_S = 15.0  # Turbo's T3 reads up to 15s of prompt (ENC_COND_LEN)
MIN_REF_S = 5.0  # Chatterbox Turbo asserts > 5s
TARGET_LUFS = -20.0
PEAK_CEILING = 0.8  # ~-2 dBFS
TAG_RE = re.compile(r"\[[^\]]+\]")
# Tags with no Chatterbox equivalent -> nearest Chatterbox tag ("" = neutral delivery)
TAG_ALIASES = {
    "[serious]": "[dramatic]",
    "[explanatory]": "[narration]",
    "[confident]": "[narration]",
    "[flat]": "",
}


def clean_reference(src: Path, dst: Path) -> float:
    """Drop long silences, keep the first MAX_REF_S of speech, normalize loudness. Returns duration."""
    wav, _ = librosa.load(src, sr=REF_SR, mono=True)

    intervals = librosa.effects.split(wav, top_db=SILENCE_TOP_DB)
    if len(intervals) == 0:
        raise ValueError("no speech detected in the recording")
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if (start - merged[-1][1]) / REF_SR < KEEP_GAP_S:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    speech = np.concatenate([wav[s:e] for s, e in merged])[: int(MAX_REF_S * REF_SR)]

    duration = len(speech) / REF_SR
    if duration <= MIN_REF_S:
        raise ValueError(f"only {duration:.1f}s of speech; record at least {MIN_REF_S + 1:.0f}s")

    speech = pyln.normalize.loudness(speech, pyln.Meter(REF_SR).integrated_loudness(speech), TARGET_LUFS)
    peak = np.abs(speech).max()
    if peak > PEAK_CEILING:
        speech *= PEAK_CEILING / peak

    sf.write(dst, speech, REF_SR)
    return duration


def normalize_tags(line: str, known: set[str]) -> tuple[str, list[str]]:
    """Map aliased tags, drop unknown ones (the model would speak them). Returns (line, dropped)."""
    line = TAG_RE.sub(lambda m: TAG_ALIASES.get(m[0].lower(), m[0]), line)
    dropped = sorted(set(TAG_RE.findall(line)) - known)
    for tag in dropped:
        line = line.replace(tag, "")
    return " ".join(line.split()), dropped


if __name__ == "__main__":
    known = {"[laugh]", "[sigh]", "[dramatic]"}
    assert normalize_tags("Hi [laugh] there", known) == ("Hi [laugh] there", [])
    assert normalize_tags("Hi [Serious] there", known) == ("Hi [dramatic] there", [])
    assert normalize_tags("Hi [flat] [wat]  there", known) == ("Hi there", ["[wat]"])
    assert normalize_tags("[wat]", known) == ("", ["[wat]"])
    print("ok")
