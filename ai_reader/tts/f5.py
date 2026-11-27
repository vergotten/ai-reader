from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np

from ..paths import download_note, models_dir
from ..stress import stresser
from ..voices.dataset import ASR_MODEL
from ..voices.library import LIB_PREFIX, VoiceLibrary
from .base import Log, TTSEngine

SAMPLE_RATE = 24_000
_ENDS_WITH_PUNCT = re.compile(r"[.!?…]$")


def base_checkpoint(model_id: str, log: Log = lambda _m: None) -> tuple[str, str]:
    """(checkpoint, vocab) files of a base model, downloading them if needed.

    f5_tts opens .pt checkpoints only in torch's safe mode, which can't read ESpeech's file. So a .pt is turned
    once into fp16 EMA weights in safetensors (models/f5/<id>.safetensors, a quarter of the size: ESpeech
    ships two fp32 copies of the weights) and the original is removed from the download cache.
    """
    from huggingface_hub import hf_hub_download

    repo, ckpt, vocab, label = F5Engine.MODELS[model_id]
    vocab_path = hf_hub_download(repo, vocab)
    if not ckpt.endswith(".pt"):
        return hf_hub_download(repo, ckpt), vocab_path
    converted = models_dir() / "f5" / f"{model_id}.safetensors"
    if not converted.is_file():
        log(f"F5-TTS: «{label}» готовится к работе (один раз: перевод в компактный формат)…")
        source = Path(hf_hub_download(repo, ckpt))
        convert_checkpoint(source, converted)
        for path in {source, source.resolve()}:  # the cache entry and the file it points to
            path.unlink(missing_ok=True)
    return str(converted), vocab_path


def convert_checkpoint(source: Path, target: Path) -> None:
    """EMA weights of an F5-TTS .pt checkpoint as fp16 safetensors (the form f5_tts loads with use_ema)."""
    import torch
    from safetensors.torch import save_file

    # torch's full unpickler: the file is from the model's official repository (see F5Engine.MODELS)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    ema = checkpoint.get("ema_model_state_dict", checkpoint)
    weights = {k: (v.half() if torch.is_floating_point(v) else v).contiguous() for k, v in ema.items()}
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    save_file(weights, str(tmp))
    os.replace(tmp, target)


class F5Engine(TTSEngine):
    id = "f5"
    name = "F5-TTS — русские дообучения (HF), клонирование голоса"
    description = (
        "Диффузионная модель F5-TTS, дообученная на русском (~1,4–2,7 ГБ). Говорит голосом из образца "
        "5–12 с: нужен WAV и его точный текст (если текст пуст, его распознает Whisper, ещё ~1,6 ГБ). "
        "Ударения расставляет RUAccent; своё ударение — «+» перед гласной: «з+амок». "
        "Нужна видеокарта: на CPU очень медленно."
    )
    languages = ("ru",)
    supports_cloning = True
    needs_reference_text = True
    stress_marks = True
    builtin_stress = True  # runs RUAccent itself
    style_controls = ("quality",)
    max_chars = 220  # reference + generated speech should stay under ~25 s, which is what F5 was trained on
    min_chunk_chars = 25
    requirements = ("f5_tts", "ruaccent", "torch")
    extra = "f5"
    license_notice = (
        "Модели F5-TTS для русского:\n"
        "• Misha24-10/F5-TTS_RUSSIAN — лицензия CC BY-NC 4.0, только некоммерческое использование;\n"
        "• ESpeech/ESpeech-TTS-1_RL-V2 — лицензия Apache 2.0.\n"
        "Клонируйте только свой голос или голос человека, который на это согласился.\n\n"
        "Принять условия и скачать модель?"
    )

    # voice id -> (HF repo, checkpoint, vocab, label)
    MODELS = {
        "misha24-v2": (
            "Misha24-10/F5-TTS_RUSSIAN",
            "F5TTS_v1_Base_v2/model_last_inference.safetensors",
            "F5TTS_v1_Base/vocab.txt",
            "F5-TTS_RUSSIAN v2 — Misha24-10 (CC BY-NC)",
        ),
        "espeech-rl-v2": (
            "ESpeech/ESpeech-TTS-1_RL-V2",
            "espeech_tts_rlv2.pt",
            "vocab.txt",
            "ESpeech-TTS-1 RL-V2 (Apache 2.0)",
        ),
    }
    MODEL_CFG = {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "conv_layers": 4}  # F5TTS_v1_Base
    MAX_REFERENCE_SECONDS = 15.0
    AUTO_TRIM_SECONDS = 12.0  # an untranscribed reference is cut to this length before Whisper hears it

    def __init__(self) -> None:
        super().__init__()
        self.sample_rate = SAMPLE_RATE
        self._infer = None  # f5_tts.infer.utils_infer
        self._vocoder = None
        self._model = None
        self._model_id: str | None = None  # cache key: base model id or path of a fine-tuned checkpoint
        self._refs: dict[tuple, tuple[object, str]] = {}
        self._log: Log = lambda _msg: None

    def voices(self, language: str) -> list[tuple[str, str]]:
        if language not in self.languages:
            return []
        base = [(voice_id, spec[3]) for voice_id, spec in self.MODELS.items()]
        mine = [(v.lib_id, v.label()) for v in VoiceLibrary().list() if v.ref_path]
        return mine + base

    def _load(self, device: str, log: Log) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pydub warns about missing ffmpeg; it is not needed for WAV
            from f5_tts.infer import utils_infer
        self._infer = utils_infer
        log("F5-TTS: загрузка вокодера Vocos…")
        self._vocoder = utils_infer.load_vocoder("vocos", device=device)
        stresser.load(log)

    def _unload(self) -> None:
        self._model = None
        self._model_id = None
        self._vocoder = None
        self._refs.clear()

    def prepare(self, voice: str, log: Log) -> None:
        with self._lock:
            self._log = log
            base_id, own_ckpt = voice, None
            if voice.startswith(LIB_PREFIX):
                v = VoiceLibrary().get(voice)
                if v is None or not v.ref_path:
                    raise RuntimeError("Голос не найден или у него нет эталонного фрагмента (окно «Мои голоса»)")
                base_id = v.base_model if v.base_model in self.MODELS else "misha24-v2"
                own_ckpt = str(v.model_path) if v.model_path else None
            key = own_ckpt or base_id
            if key == self._model_id:
                return
            if base_id not in self.MODELS:
                raise RuntimeError(f"Неизвестная модель F5-TTS: {voice}")
            from f5_tts.model import DiT

            repo, _ckpt, _vocab, label = self.MODELS[base_id]
            self._model = None
            self._model_id = None
            log(f"F5-TTS: загрузка «{label}»{download_note(repo, '1–2,5 ГБ')}…")
            base_path, vocab_path = base_checkpoint(base_id, log)
            ckpt_path = own_ckpt or base_path
            if own_ckpt:
                log(f"F5-TTS: дообученный голос {own_ckpt}")
            self._model = self._infer.load_model(DiT, self.MODEL_CFG, ckpt_path, vocab_file=vocab_path, device=self.device)
            self._model_id = key

    def forget_voices(self) -> None:
        with self._lock:
            self._model_id = None  # prepare() reloads the checkpoint next time
            self._refs.clear()

    def stress(self, text: str) -> str:
        """Put '+' before stressed vowels (and restore «ё»), keeping stresses the user typed."""
        return stresser.process(text)

    def _reference(self, path: str, transcript: str | None):
        """(mono 24 kHz torch tensor [1, n], stressed transcript), cached per file and transcript."""
        import soundfile as sf
        import soxr
        import torch

        from ..text import normalize

        transcript = (transcript or "").strip()
        key = (path, os.path.getmtime(path), transcript)
        if key in self._refs:
            return self._refs[key]
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = soxr.resample(audio, sr, SAMPLE_RATE).astype(np.float32)
        audio = _trim_silence(audio)
        seconds = len(audio) / SAMPLE_RATE
        if seconds < 1.0:
            raise RuntimeError("Образец голоса пустой или слишком короткий (нужно 5–12 с речи)")
        if not transcript:
            audio = _cut_at_pause(audio, self.AUTO_TRIM_SECONDS)
            transcript = self._transcribe(audio)
        elif seconds > self.MAX_REFERENCE_SECONDS:
            raise RuntimeError(
                f"Образец голоса длится {seconds:.0f} с — обрежьте его до 5–12 с (и текст образца вместе с ним)"
            )
        text = normalize(transcript, "ru")
        if not _ENDS_WITH_PUNCT.search(text):
            text += "."
        text = self.stress(text) + " "
        audio = np.concatenate([audio, np.zeros(SAMPLE_RATE // 20, dtype=np.float32)])  # as f5_tts does: +50 ms
        ref = (torch.from_numpy(audio).unsqueeze(0), text)
        self._refs[key] = ref
        return ref

    def _transcribe(self, audio: np.ndarray) -> str:
        self._log(f"F5-TTS: текст образца не задан — распознаю его Whisper{download_note(ASR_MODEL, '~1,6 ГБ')}…")
        infer = self._infer
        infer.initialize_asr_pipeline(device=self.device)
        try:
            text = infer.asr_pipe(
                {"raw": audio, "sampling_rate": SAMPLE_RATE},
                generate_kwargs={"task": "transcribe", "language": "russian"},
                return_timestamps=False,
            )["text"].strip()
        finally:
            infer.asr_pipe = None  # free VRAM for the TTS model
            from .base import free_torch_memory

            free_torch_memory()
        if not text:
            raise RuntimeError("Не удалось распознать речь в образце голоса — впишите его текст вручную")
        self._log(f"F5-TTS: текст образца: «{text}». Если он неточен, впишите правильный в поле «Текст образца».")
        return text

    def _synthesize(self, text, voice, speed, language, speaker_wav, speaker_text, style) -> np.ndarray:
        if voice.startswith(LIB_PREFIX):
            v = VoiceLibrary().get(voice)
            if v is None or not v.ref_path:
                raise RuntimeError("У голоса нет эталонного фрагмента")
            speaker_wav, speaker_text = str(v.ref_path), v.reference_text
        if not speaker_wav or not os.path.isfile(speaker_wav):
            raise RuntimeError("Для F5-TTS выберите WAV-файл с образцом голоса (5–12 с)")
        ref_audio, ref_text = self._reference(speaker_wav, speaker_text)
        gen_text = self.stress(text)
        ref_seconds = ref_audio.shape[-1] / SAMPLE_RATE
        # same budget as f5_tts.infer_process: keep reference + generated audio within ~25 s
        max_bytes = int(len(ref_text.encode("utf-8")) / ref_seconds * max(25 - ref_seconds, 5) * speed)
        batches = self._infer.chunk_text(gen_text, max_chars=max_bytes)
        wave, _sr, _spec = next(
            self._infer.infer_batch_process(
                (ref_audio, SAMPLE_RATE),
                ref_text,
                batches,
                self._model,
                self._vocoder,
                progress=None,
                nfe_step=style.quality,
                speed=speed,
                device=self.device,
            )
        )
        if wave is None:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(wave, dtype=np.float32)


def _frame_db(audio: np.ndarray, frame: int) -> np.ndarray:
    n = len(audio) // frame
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    return 20 * np.log10(rms)


def _trim_silence(audio: np.ndarray, threshold_db: float = -42.0) -> np.ndarray:
    frame = SAMPLE_RATE // 100
    loud = np.flatnonzero(_frame_db(audio, frame) > threshold_db)
    if loud.size == 0:
        return audio[:0]
    start = max(0, loud[0] - 5) * frame  # keep 50 ms around the speech
    end = min(len(audio), (loud[-1] + 6) * frame)
    return audio[start:end]


def _cut_at_pause(audio: np.ndarray, max_seconds: float) -> np.ndarray:
    """Shorten to at most max_seconds, cutting at the quietest point of the last few seconds."""
    limit = int(max_seconds * SAMPLE_RATE)
    if len(audio) <= limit:
        return audio
    frame = SAMPLE_RATE // 20
    db = _frame_db(audio[:limit], frame)
    lo = int(len(db) * 0.6)
    cut = (lo + int(np.argmin(db[lo:]))) * frame
    return audio[:cut]
