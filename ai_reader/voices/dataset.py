"""Turn a pile of voice recordings into a training set: cut at pauses, transcribe with Whisper, add stress marks."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .audio_io import load_audio
from .library import Clip, Voice

SAMPLE_RATE = 24_000
ASR_RATE = 16_000
ASR_MODEL = "openai/whisper-large-v3-turbo"
FRAME = SAMPLE_RATE // 100  # 10 ms

MIN_CLIP = 2.0  # seconds
MAX_CLIP = 12.0  # F5-TTS fine-tunes best on 3–15 s clips; the reference + generated text must fit in ~25 s
SENTENCE_PAUSE = 0.55  # a pause this long ends a clip once it is at least MIN_CLIP long
MIN_PAUSE = 0.18  # shorter gaps are not considered for cutting
PAD = 0.12

# Whisper is known to "hear" these in silence or music
_HALLUCINATIONS = re.compile(
    r"субтитр|продолжение следует|спасибо за просмотр|подпишитесь|редактор|корректор|dimatorzok|amara\.org",
    re.IGNORECASE,
)

Log = Callable[[str], None]
Progress = Callable[[int, int, str], None]


@dataclass
class _Segment:
    start: int  # samples
    end: int


def _frame_db(audio: np.ndarray) -> np.ndarray:
    n = len(audio) // FRAME
    if n == 0:
        return np.full(1, -100.0)
    frames = audio[: n * FRAME].reshape(n, FRAME)
    return 20 * np.log10(np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12))


def normalize_loudness(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    db = _frame_db(audio)
    speech = db[db > np.percentile(db, 50)] if len(db) > 10 else db
    level = 10 * np.log10(np.mean(10 ** (speech / 10)) + 1e-12)
    gain = 10 ** ((target_db - level) / 20)
    out = audio * min(gain, 30.0)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.97:
        out *= 0.97 / peak
    return out.astype(np.float32)


def split_on_pauses(audio: np.ndarray, min_len: float = MIN_CLIP, max_len: float = MAX_CLIP) -> list[tuple[int, int]]:
    """Sample ranges of speech, 2–12 s each, cut in the middle of pauses (sentence pauses preferred)."""
    db = _frame_db(audio)
    if len(db) < 10:
        return []
    floor, loud = np.percentile(db, 10), np.percentile(db, 95)
    if loud - floor < 10:  # no dynamics: noise or constant tone
        return []
    threshold = floor + max(6.0, 0.3 * (loud - floor))
    silent = db < threshold
    speech_frames = np.flatnonzero(~silent)
    first, last = int(speech_frames[0]), int(speech_frames[-1]) + 1

    # pauses inside [first, last): (start_frame, end_frame)
    pauses: list[tuple[int, int]] = []
    i = first
    min_pause = int(MIN_PAUSE * 100)
    while i < last:
        if silent[i]:
            j = i
            while j < last and silent[j]:
                j += 1
            if j - i >= min_pause:
                pauses.append((i, j))
            i = j
        else:
            i += 1

    min_f, max_f, sentence_f = int(min_len * 100), int(max_len * 100), int(SENTENCE_PAUSE * 100)
    segments: list[tuple[int, int]] = []
    start = first
    candidates: list[tuple[int, int]] = []
    for p in pauses + [(last, last)]:
        while p[0] - start > max_f:  # too long: cut at the best earlier pause
            good = [c for c in candidates if c[0] - start >= min_f]
            if good:
                c = max(good, key=lambda c: (c[1] - c[0]) + 0.05 * (c[0] - start))
                segments.append((start, c[0]))
                start = c[1]
                candidates = [x for x in candidates if x[0] > c[0]]
            else:  # one long breath with no pause: hard cut
                segments.append((start, start + max_f))
                start += max_f
                candidates = []
        if p[0] == last:
            if p[0] - start >= min_f // 2:
                segments.append((start, last))
            break
        if p[0] - start >= min_f and p[1] - p[0] >= sentence_f:
            segments.append((start, p[0]))
            start = p[1]
            candidates = []
        else:
            candidates.append(p)

    pad = int(PAD * 100)
    out = []
    for k, (s, e) in enumerate(segments):
        lo = segments[k - 1][1] if k else 0
        hi = segments[k + 1][0] if k + 1 < len(segments) else len(db)
        s, e = max(lo, s - pad), min(hi, e + pad)
        if (e - s) / 100 >= 1.0:
            out.append((s * FRAME, min(len(audio), e * FRAME)))
    return out


def prepare_dataset(
    voice: Voice,
    files: list[str],
    device_preference: str,
    log: Log,
    progress: Progress,
    cancelled: Callable[[], bool],
    add_stress: bool = True,
) -> int:
    """Cut recordings into clips in voice.clips_dir, transcribe and stress them. Returns the number of new clips."""
    import soundfile as sf

    from ..pipeline import Cancelled

    voice.clips_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(m.group(1)) for c in voice.clips if (m := re.search(r"(\d+)\.wav$", c.file))]
    number = max(existing, default=0)
    new: list[tuple[Clip, np.ndarray]] = []
    read_errors: list[str] = []

    for n, path in enumerate(files, 1):
        if cancelled():
            raise Cancelled
        progress(n - 1, len(files), f"Нарезка: {os.path.basename(path)}")
        try:
            audio = load_audio(path, SAMPLE_RATE)
        except RuntimeError as e:
            message = f"{os.path.basename(path)}: {e}"
            read_errors.append(message)
            log(message)
            continue
        if len(audio) < SAMPLE_RATE:
            log(f"Пропуск (короче секунды): {os.path.basename(path)}")
            continue
        audio = normalize_loudness(audio)
        spans = split_on_pauses(audio)
        log(f"{os.path.basename(path)}: {len(audio) / SAMPLE_RATE:.0f} с → {len(spans)} фрагм.")
        for s, e in spans:
            number += 1
            rel = f"clips/{number:05d}.wav"
            piece = audio[s:e]
            sf.write(voice.path(rel), piece, SAMPLE_RATE, subtype="PCM_16")
            new.append((Clip(rel, "", round(len(piece) / SAMPLE_RATE, 2), True), piece))

    if not new:
        if read_errors:
            details = "\n".join(read_errors[:5])
            extra = f"\n…и ещё {len(read_errors) - 5} файл(ов)." if len(read_errors) > 5 else ""
            raise RuntimeError(f"Не удалось прочитать ни одной записи:\n{details}{extra}")
        raise RuntimeError("В записях не найдено речи. Проверьте, что в файлах слышна речь без длинных пауз.")

    _transcribe(new, device_preference, log, progress, cancelled)

    if add_stress:
        from ..stress import stresser

        if stresser.available():
            stresser.load(log)
            for k, (clip, _a) in enumerate(new):
                if clip.text:
                    clip.text = stresser.process(clip.text)
                if k % 20 == 0:
                    progress(k, len(new), "Ударения")
        else:
            log("RUAccent не установлен — ударения в текстах не расставлены.")

    voice.clips.extend(clip for clip, _a in new)
    if not voice.ref_path:
        voice.pick_reference()
    voice.save()
    ok = sum(1 for c, _a in new if c.enabled)
    log(f"Готово: {len(new)} фрагментов, {ok} годных, {sum(c.duration for c, _a in new if c.enabled) / 60:.1f} мин речи.")
    return len(new)


def _transcribe(items: list[tuple[Clip, np.ndarray]], device_preference: str, log: Log, progress: Progress, cancelled) -> None:
    import soxr
    import torch
    from transformers import pipeline

    from ..pipeline import Cancelled
    from ..text import normalize
    from ..tts.base import free_torch_memory, resolve_device

    device = resolve_device(device_preference)
    dtype = torch.float16 if device == "cuda" and torch.cuda.get_device_properties(0).major >= 7 else torch.float32
    log(f"Whisper: загрузка {ASR_MODEL} ({device}, первый раз скачивается ~1,6 ГБ)…")
    asr = pipeline("automatic-speech-recognition", model=ASR_MODEL, torch_dtype=dtype, device=device)
    batch = 8 if device == "cuda" else 2
    try:
        for i in range(0, len(items), batch):
            if cancelled():
                raise Cancelled
            progress(i, len(items), "Распознавание речи (Whisper)")
            group = items[i : i + batch]
            inputs = [soxr.resample(a, SAMPLE_RATE, ASR_RATE).astype(np.float32) for _c, a in group]
            results = asr(
                inputs,
                batch_size=len(inputs),
                generate_kwargs={"language": "russian", "task": "transcribe"},
                return_timestamps=False,
            )
            for (clip, _a), result in zip(group, results, strict=True):
                raw = (result.get("text") or "").strip()
                text = normalize(raw, "ru") if raw else ""
                if text and not re.search(r"[.!?…]$", text):
                    text += "."
                clip.text = text
                clip.enabled = _plausible(text, clip.duration)
    finally:
        del asr
        free_torch_memory()


def _plausible(text: str, duration: float) -> bool:
    letters = len(re.findall(r"[а-яёА-ЯЁ]", text))
    if letters < 3 or _HALLUCINATIONS.search(text):
        return False
    rate = letters / max(duration, 0.1)
    return 4.0 <= rate <= 22.0
