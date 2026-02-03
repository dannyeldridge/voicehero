"""Whisper transcription functionality."""

import time
from typing import TYPE_CHECKING, Literal

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

        start_time = time.time()

        console.print("[dim](First run may take 10-30s to download model from Hugging Face)[/dim]")

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

        elapsed = time.time() - start_time
        console.print(f"[green]✓ Model loaded in {elapsed:.1f}s[/green]\n")

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

    def dispose(self) -> None:
        """Clean up resources."""
        if self.model:
            del self.model
            self.model = None
