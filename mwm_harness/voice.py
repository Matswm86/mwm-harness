"""Speech to text for the panel's mic button: faster-whisper on this machine, no network.

Optional: install with ``pip install -e .[voice]``. The model loads on the first
clip and is dropped after ``IDLE_SECONDS`` without one, so the GPU memory goes
back to the local LLM. CUDA first (the cuBLAS and cuDNN wheels from the extra
are loaded by path, no LD_LIBRARY_PATH needed); if that fails, a smaller model
runs on the CPU.
"""

from __future__ import annotations

import asyncio
import ctypes
import glob
import io
import site
import time
from dataclasses import dataclass
from typing import Any

IDLE_SECONDS = 600.0
CPU_MODEL = "small"
CUDA_LIBS = (
    "cublas/lib/libcublasLt.so.12",
    "cublas/lib/libcublas.so.12",
    "cudnn/lib/libcudnn.so.9",
)


@dataclass
class Transcript:
    text: str = ""
    language: str = ""
    seconds: float = 0.0
    model: str = ""
    error: str = ""


def _load_cuda_libs() -> None:
    for lib in CUDA_LIBS:
        for root in site.getsitepackages():
            for path in glob.glob(f"{root}/nvidia/{lib}"):
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


class Transcriber:
    def __init__(self, model: str = "medium", device: str = "auto", language: str = "") -> None:
        self.model_name = model
        self.device = device
        self.language = language or None
        self._model: Any = None
        self._label = ""
        self._used = 0.0
        self._lock = asyncio.Lock()
        self._reaper: asyncio.Task[None] | None = None

    def _load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("voice needs faster-whisper: pip install -e .[voice]") from exc
        attempts = []
        if self.device in ("auto", "cuda"):
            attempts.append((self.model_name, "cuda", "int8_float16"))
        if self.device in ("auto", "cpu"):
            attempts.append((self.model_name if self.device == "cpu" else CPU_MODEL, "cpu", "int8"))
        errors = []
        for name, device, compute in attempts:
            try:
                if device == "cuda":
                    _load_cuda_libs()
                self._model = WhisperModel(name, device=device, compute_type=compute)
                self._label = f"{name} on {device}"
                return
            except (RuntimeError, OSError, ValueError) as exc:
                errors.append(f"{name}/{device}: {exc}")
        raise RuntimeError("no Whisper model could load: " + "; ".join(errors))

    def _run(self, audio: bytes) -> Transcript:
        started = time.monotonic()
        if self._model is None:
            self._load()
        segments, info = self._model.transcribe(
            io.BytesIO(audio), language=self.language, vad_filter=True, beam_size=1
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return Transcript(text, info.language, round(time.monotonic() - started, 2), self._label)

    async def transcribe(self, audio: bytes) -> Transcript:
        async with self._lock:  # one clip at a time: the model is not shared between threads
            try:
                result = await asyncio.to_thread(self._run, audio)
            except (
                Exception
            ) as exc:  # decoder and CUDA errors come in many types; all go to the page
                return Transcript(error=f"{type(exc).__name__}: {exc}")
            self._used = time.monotonic()
            if self._reaper is None or self._reaper.done():
                self._reaper = asyncio.create_task(self._drop_when_idle())
            return result

    async def warm(self) -> None:
        """Load the model while the person is still speaking, so the clip is not kept waiting."""
        async with self._lock:
            if self._model is not None:
                return
            try:
                await asyncio.to_thread(self._load)
            except RuntimeError:
                return  # the clip itself reports the error when it arrives
            self._used = time.monotonic()
            if self._reaper is None or self._reaper.done():
                self._reaper = asyncio.create_task(self._drop_when_idle())

    async def _drop_when_idle(self) -> None:
        while self._model is not None:
            await asyncio.sleep(30)
            if time.monotonic() - self._used > IDLE_SECONDS and not self._lock.locked():
                self._model = None

    async def close(self) -> None:
        if self._reaper:
            self._reaper.cancel()
        self._model = None
