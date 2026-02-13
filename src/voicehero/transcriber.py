"""Whisper transcription functionality."""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from rich.console import Console

from .logger import get_logger

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

console = Console()

ModelSize = Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]


class AudioTranscriber:
    """Transcribes audio using Whisper."""

    def __init__(
        self,
        model_size: ModelSize = "base",
        device: str = "auto",
        compute_type: str = "auto",
    ):
        """Initialize the transcriber.

        Args:
            model_size: Size of the Whisper model to use
            device: Device to run on ("cpu", "cuda", or "auto")
            compute_type: Computation type ("int8", "float16", "float32", or "auto")
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: WhisperModel | None = None

    def initialize(self) -> None:
        """Load the Whisper model."""
        console.print(f"[cyan]Initializing Whisper ({self.model_size})...[/cyan]")

        from faster_whisper import WhisperModel
        from faster_whisper.utils import _MODELS

        start_time = time.time()

        # Pre-download with visible progress if not already cached.
        # faster_whisper silences tqdm internally (disabled_tqdm), so we
        # call huggingface_hub.snapshot_download ourselves first.
        model_path = self._ensure_model_downloaded(self.model_size, _MODELS)

        self.model = WhisperModel(
            model_path,
            device=self.device,
            compute_type=self.compute_type,
        )

        elapsed = time.time() - start_time
        console.print(f"[green]✓ Model loaded in {elapsed:.1f}s[/green]\n")

    @staticmethod
    def _ensure_model_downloaded(model_size: str, models_map: dict[str, str]) -> str:
        """Download the model with visible progress if not cached, return its path."""
        import huggingface_hub

        repo_id = models_map.get(model_size)
        if repo_id is None:
            # Not a known size name — treat as a direct repo ID or local path
            return model_size

        allow_patterns = [
            "config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.*",
        ]

        # Check if already cached (fast, no network)
        try:
            path = huggingface_hub.snapshot_download(
                repo_id,
                allow_patterns=allow_patterns,
                local_files_only=True,
            )
            return path
        except FileNotFoundError:
            pass

        # Not cached — download with progress visible
        console.print(f"[dim]Downloading {repo_id} from Hugging Face (first time only)...[/dim]")
        path = huggingface_hub.snapshot_download(
            repo_id,
            allow_patterns=allow_patterns,
        )
        return path

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            language: Language code (e.g., "en" for English)

        Returns:
            Transcribed text
        """
        logger = get_logger()
        logger.info("=== TRANSCRIPTION START ===")

        if self.model is None:
            logger.error("Model not initialized")
            raise RuntimeError("Model not initialized. Call initialize() first.")

        if len(audio) == 0:
            logger.warning("Empty audio data provided")
            return ""

        logger.debug(f"Audio input: shape={audio.shape}, dtype={audio.dtype}, length={len(audio)}")

        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            logger.debug(f"Converting audio from {audio.dtype} to float32")
            audio = audio.astype(np.float32)

        # Normalize audio to [-1, 1] range if needed
        max_val = np.abs(audio).max()
        logger.debug(f"Audio max value: {max_val}")
        if max_val > 1.0:
            logger.debug(f"Normalizing audio (max_val={max_val})")
            audio = audio / max_val

        # Transcribe
        logger.info(f"Starting Whisper transcription (model={self.model_size}, language={language})")
        start_time = time.time()

        try:
            segments, info = self.model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                ),
            )

            # Combine all segments into a single text
            text = " ".join(segment.text.strip() for segment in segments)
            text = text.strip()

            elapsed = time.time() - start_time
            logger.info(f"Transcription completed in {elapsed:.2f}s: {len(text)} chars, {len(text.split())} words")
            logger.debug(f"Transcribed text: {text[:100]}..." if len(text) > 100 else f"Transcribed text: {text}")

            return text
        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception(f"Transcription failed after {elapsed:.2f}s: {e}")
            raise

    def transcribe_file_with_progress(
        self,
        file_path: str | Path,
        language: str = "en",
        progress_callback: Callable[[float, float], None] | None = None,
    ) -> str:
        """Transcribe an audio file to text with progress reporting.

        Args:
            file_path: Path to the audio file (wav, mp3, m4a, ogg, flac, etc.)
            language: Language code
            progress_callback: Called with (current_seconds, total_seconds) as segments complete

        Returns:
            Transcribed text
        """
        logger = get_logger()
        logger.info(f"=== FILE TRANSCRIPTION START: {file_path} ===")

        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        segments_gen, info = self.model.transcribe(
            str(file_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        total_duration = info.duration
        logger.info(f"Audio duration: {total_duration:.1f}s")

        parts: list[str] = []
        for segment in segments_gen:
            text = segment.text.strip()
            if text:
                parts.append(text)
            if progress_callback is not None:
                progress_callback(segment.end, total_duration)

        result = " ".join(parts).strip()
        logger.info(f"File transcription completed: {len(result)} chars, {len(result.split())} words")
        return result

    def dispose(self) -> None:
        """Clean up resources."""
        if self.model:
            del self.model
            self.model = None
