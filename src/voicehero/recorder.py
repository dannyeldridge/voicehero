"""Audio recording functionality."""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from rich.console import Console

console = Console()


class AudioRecorder:
    """Records audio from the microphone."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
        debug: bool = False,
    ):
        """Initialize the audio recorder.

        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono)
            dtype: Data type for audio samples
            debug: Enable debug mode with extra logging
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.debug = debug
        self.recording = False
        self.audio_data: list[np.ndarray] = []
        self.stream: Optional[sd.InputStream] = None
        self.start_time: Optional[float] = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback for audio stream to capture audio data."""
        if status and self.debug:
            console.print(f"[yellow]Audio status: {status}[/yellow]")

        if self.recording:
            self.audio_data.append(indata.copy())

    def start(self) -> None:
        """Start recording audio."""
        if self.recording:
            raise RuntimeError("Recording already in progress")

        self.audio_data = []
        self.recording = True
        self.start_time = time.time()

        # Create and start the audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._audio_callback,
        )
        self.stream.start()

        if self.debug:
            console.print(f"[green]Recording started at {self.sample_rate}Hz[/green]")

    def stop(self) -> np.ndarray:
        """Stop recording and return the audio data.

        Returns:
            Audio data as a numpy array
        """
        if not self.recording:
            raise RuntimeError("No recording in progress")

        self.recording = False

        # Stop and close the stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Combine all audio chunks
        if not self.audio_data:
            return np.array([], dtype=self.dtype)

        audio = np.concatenate(self.audio_data, axis=0)

        # Flatten to 1D if it's a multi-channel recording
        if audio.ndim > 1:
            audio = audio.flatten()

        if self.debug and self.start_time:
            duration = time.time() - self.start_time
            samples = len(audio)
            console.print(f"[blue]Recording stopped: {samples} samples, {duration:.2f}s[/blue]")

        return audio

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.recording

    def save_debug_recording(self, audio: np.ndarray, recordings_dir: Path) -> None:
        """Save a debug recording to disk.

        Args:
            audio: Audio data to save
            recordings_dir: Directory to save recordings
        """
        try:
            from scipy.io import wavfile

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = recordings_dir / f"recording-{timestamp}.wav"

            # Convert float32 to int16 for WAV file
            if audio.dtype == np.float32:
                audio_int16 = (audio * 32767).astype(np.int16)
            else:
                audio_int16 = audio

            wavfile.write(filename, self.sample_rate, audio_int16)
            console.print(f"[dim]Debug: Saved to {filename}[/dim]")
        except ImportError:
            console.print("[yellow]scipy not installed, skipping debug save[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Failed to save debug recording: {e}[/yellow]")
