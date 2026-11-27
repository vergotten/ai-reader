"""User voices: recordings cut into clips with transcripts, a reference clip, and optionally a fine-tuned F5 model.

Layout of one voice (under paths.voices_dir()):

    <id>/voice.json      name, reference clip, base model, trained model
    <id>/clips.json      [{file, text, duration, enabled}]
    <id>/clips/0001.wav  24 kHz mono clips, 2–12 s each
    <id>/train/          F5-TTS checkpoints while fine-tuning (large; can be deleted after export)
    <id>/model.pt        fine-tuned weights used for synthesis (EMA, fp16)
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..paths import voices_dir

LIB_PREFIX = "lib:"
DEFAULT_VOICE = "vovan"  # the fine-tuned voice shipped with the full installer; F5-TTS reads with it by default


def bundled_voices_dir() -> Path:
    """Voices shipped with the app: <app>/voices, filled by the installer (absent in the lite build)."""
    return Path(__file__).resolve().parents[2] / "voices"


def install_bundled_voices(source: Path | None = None, target: Path | None = None) -> list[str]:
    """Copy the shipped voices into the user's library once. A voice the user already has is left alone,
    so retraining it, or editing its clips, survives app updates. Returns the ids copied now."""
    source = source or bundled_voices_dir()
    target = target or voices_dir()
    copied = []
    if not source.is_dir():
        return copied
    for d in sorted(source.iterdir()):
        if (d / "voice.json").is_file() and not (target / d.name).exists():
            tmp = target / (d.name + ".part")
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(d, tmp)
            tmp.rename(target / d.name)  # a half-copied voice never looks like a real one
            copied.append(d.name)
    return copied
DEFAULT_BASE_MODEL = "misha24-v2"

_TRANSLIT = dict(
    zip(
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
        ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u", "f", "h", "ts", "ch", "sh", "sch", "_", "y", "_", "e", "yu", "ya"],
        strict=True,
    )
)


def _slug(name: str) -> str:
    s = "".join(_TRANSLIT.get(c, c) for c in name.lower())
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-_")
    return s[:40] or "voice"


@dataclass
class Clip:
    file: str  # relative to the voice dir
    text: str
    duration: float
    enabled: bool = True


@dataclass
class Voice:
    id: str
    name: str
    dir: Path
    reference: str | None = None  # relative path of the reference clip
    reference_text: str = ""
    base_model: str = DEFAULT_BASE_MODEL
    model: str | None = None  # relative path of the fine-tuned checkpoint
    trained_updates: int = 0
    trained_seconds: float = 0.0
    created: str = ""
    clips: list[Clip] = field(default_factory=list)

    # ------------------------------------------------------------------ paths

    def path(self, rel: str) -> Path:
        return self.dir / rel

    @property
    def clips_dir(self) -> Path:
        return self.dir / "clips"

    @property
    def train_dir(self) -> Path:
        return self.dir / "train"

    @property
    def ref_path(self) -> Path | None:
        if self.reference and self.path(self.reference).is_file():
            return self.path(self.reference)
        return None

    @property
    def model_path(self) -> Path | None:
        if self.model and self.path(self.model).is_file():
            return self.path(self.model)
        return None

    @property
    def is_trained(self) -> bool:
        return self.model_path is not None

    @property
    def lib_id(self) -> str:
        return LIB_PREFIX + self.id

    def enabled_clips(self) -> list[Clip]:
        return [c for c in self.clips if c.enabled and c.text.strip() and self.path(c.file).is_file()]

    def total_seconds(self, enabled_only: bool = True) -> float:
        clips = self.enabled_clips() if enabled_only else self.clips
        return sum(c.duration for c in clips)

    def label(self) -> str:
        kind = "дообучен" if self.is_trained else "клон по образцу"
        return f"★ {self.name} ({kind})"

    # ------------------------------------------------------------------ persistence

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = asdict(self)
        for key in ("dir", "clips"):
            meta.pop(key)
        _write_json(self.dir / "voice.json", meta)
        _write_json(self.dir / "clips.json", [asdict(c) for c in self.clips])

    @classmethod
    def load(cls, directory: Path) -> Voice:
        meta = json.loads((directory / "voice.json").read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__ if f not in ("dir", "clips")}
        voice = cls(dir=directory, **{k: v for k, v in meta.items() if k in known})
        clips_file = directory / "clips.json"
        if clips_file.is_file():
            voice.clips = [Clip(**c) for c in json.loads(clips_file.read_text(encoding="utf-8"))]
        return voice

    def pick_reference(self) -> Clip | None:
        """Choose a 5–11 s clip whose transcript looks complete; F5 needs one as a prompt."""
        candidates = [c for c in self.enabled_clips() if 4.0 <= c.duration <= 11.5]
        if not candidates:
            candidates = self.enabled_clips()
        if not candidates:
            return None

        def score(c: Clip) -> float:
            s = abs(c.duration - 8.0)
            if not re.search(r"[.!?…]\s*$", c.text):
                s += 3
            return s

        best = min(candidates, key=score)
        self.reference = best.file
        self.reference_text = best.text
        return best


class VoiceLibrary:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or voices_dir()

    def list(self) -> list[Voice]:
        voices = []
        if not self.root.is_dir():
            return voices
        for d in sorted(self.root.iterdir()):
            if (d / "voice.json").is_file():
                try:
                    voices.append(Voice.load(d))
                except (OSError, ValueError, TypeError):
                    continue
        return sorted(voices, key=lambda v: v.created)

    def get(self, voice_id: str) -> Voice | None:
        voice_id = voice_id.removeprefix(LIB_PREFIX)
        d = self.root / voice_id
        if (d / "voice.json").is_file():
            return Voice.load(d)
        return None

    def create(self, name: str) -> Voice:
        base = _slug(name)
        vid, n = base, 2
        while (self.root / vid).exists():
            vid, n = f"{base}-{n}", n + 1
        voice = Voice(id=vid, name=name.strip() or vid, dir=self.root / vid, created=time.strftime("%Y-%m-%d %H:%M:%S"))
        voice.clips_dir.mkdir(parents=True, exist_ok=True)
        voice.save()
        return voice

    def delete(self, voice: Voice) -> None:
        shutil.rmtree(voice.dir, ignore_errors=True)


def is_library_voice(voice_id: str | None) -> bool:
    return bool(voice_id) and voice_id.startswith(LIB_PREFIX)


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
