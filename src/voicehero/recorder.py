"""Audio recording functionality."""

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from rich.console import Console

from .logger import get_logger

console = Console()

STREAM_TIMEOUT = 2.0  # seconds to wait for stream operations


def _run_with_timeout(func, timeout: float, description: str = "operation") -> bool:
    """Run a function with a timeout.

    Args:
        func: Function to run
        timeout: Timeout in seconds
        description: Description for error messages

    Returns:
        True if completed successfully, False if timed out
    """
    result = {"completed": False, "error": None}

    def wrapper():
        try:
            logger = get_logger()
            logger.debug(f"Starting: {description}")
            func()
            result["completed"] = True
            logger.debug(f"Completed: {description}")
        except Exception as e:
            result["error"] = e
            logger.error(f"Error in {description}: {type(e).__name__}: {e}")

    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if not result["completed"]:
        logger = get_logger()
        if thread.is_alive():
            console.print(f"[yellow]Warning: {description} timed out[/yellow]")
            logger.warning(f"TIMEOUT: {description} timed out after {timeout}s (thread still alive)")
            return False
        elif result["error"]:
            console.print(f"[yellow]Warning: {description} failed: {result['error']}[/yellow]")
            logger.error(f"FAILED: {description} - {result['error']}")
            return False
    return True


def is_bluetooth_device(device_name: str) -> bool:
    """Check if a device name indicates a Bluetooth device.

    Args:
        device_name: Name of the audio device

    Returns:
        True if the device appears to be Bluetooth
    """
    bluetooth_indicators = [
        'airpods',
        'bluetooth',
        'bt',
        'wireless',
        'beats',
        'bose',
        'sony',
        'jabra',
        'sennheiser',
    ]
    device_lower = device_name.lower()
    return any(indicator in device_lower for indicator in bluetooth_indicators)


def get_default_input_device(force_refresh: bool = False) -> tuple[int | None, dict]:
    """Get the current default input device.

    Args:
        force_refresh: If True, reinitialize PortAudio to detect new devices
                       (e.g. Bluetooth connections). Skipped by default to avoid
                       disrupting CoreAudio for built-in devices.

    Returns:
        Tuple of (device_index, device_info_dict)
    """
    try:
        if force_refresh:
            sd._terminate()
            time.sleep(0.1)
            sd._initialize()

        device_info = sd.query_devices(kind='input')
        device_index = sd.default.device[0]  # Input device index
        return device_index, device_info
    except Exception:
        return None, {}


class AudioRecorder:
    """Records audio from the microphone."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
        debug: bool = False,
        activated_bluetooth_device: Optional[str] = None,
    ):
        """Initialize the audio recorder.

        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono)
            dtype: Data type for audio samples
            debug: Enable debug mode with extra logging
            activated_bluetooth_device: Name of previously activated Bluetooth device (if any)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.debug = debug
        self.recording = False
        self.audio_data: list[np.ndarray] = []
        self.stream: Optional[sd.InputStream] = None
        self.start_time: Optional[float] = None
        self.activated_bluetooth_device = activated_bluetooth_device
        self.current_device_index: int | None = None
        self.current_device_name: str = ""

        # The audio stream is persistent: opened once and kept running for the
        # process lifetime. Recording is just a flag the callback honors. This
        # serializes the (rare) open/reopen/close operations so we never tear
        # PortAudio state out from under an in-flight call.
        self._lock = threading.Lock()
        self._stream_device_index: int | None = None  # device the open stream uses

        # Silence/dropout tracking
        self._silence_threshold: float = 1e-4  # RMS below this = silent buffer
        self._consecutive_silent_buffers: int = 0
        self._max_consecutive_silent: int = 0
        self._total_silent_buffers: int = 0
        self._total_buffers: int = 0
        self._status_errors: list[str] = []
        self._silence_onset_buffer: int | None = None  # buffer index where current silence started

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback for audio stream to capture audio data."""
        if status:
            logger = get_logger()
            status_str = str(status)
            logger.warning(f"Audio callback status: {status_str}")
            self._status_errors.append(status_str)
            if self.debug:
                console.print(f"[yellow]Audio status: {status_str}[/yellow]")

        if self.recording:
            self.audio_data.append(indata.copy())
            self._total_buffers += 1

            # Silence detection
            rms = np.sqrt(np.mean(indata ** 2))
            if rms < self._silence_threshold:
                self._consecutive_silent_buffers += 1
                self._total_silent_buffers += 1
                if self._silence_onset_buffer is None:
                    self._silence_onset_buffer = self._total_buffers - 1
                if self._consecutive_silent_buffers > self._max_consecutive_silent:
                    self._max_consecutive_silent = self._consecutive_silent_buffers
            else:
                self._consecutive_silent_buffers = 0
                self._silence_onset_buffer = None

    def _open_stream(self, device_index: int | None, device_name: str) -> None:
        """Create and start the persistent input stream. Raises on failure.

        Must be called under self._lock with no stream currently open.
        """
        logger = get_logger()
        logger.debug(f"Opening stream: device={device_index} ({device_name})")
        stream = sd.InputStream(
            device=device_index,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._audio_callback,
        )
        stream.start()
        self.stream = stream
        self._stream_device_index = device_index
        self.current_device_index = device_index
        self.current_device_name = device_name
        logger.info(f"Stream started on {device_name}")

    def _close_stream(self) -> None:
        """Stop and close the persistent stream. Must be called under self._lock.

        This is a deliberate, rare operation (device switch or shutdown) — never
        part of a normal record/stop cycle. Because it runs under the lock with no
        other PortAudio call in flight, the reinit-while-stuck race that previously
        wedged the subsystem cannot occur, so we never reinitialize PortAudio here.
        """
        if not self.stream:
            return
        logger = get_logger()
        stream = self.stream
        self.stream = None
        self._stream_device_index = None
        _run_with_timeout(stream.stop, STREAM_TIMEOUT, "stream stop")
        _run_with_timeout(stream.close, STREAM_TIMEOUT, "stream close")
        logger.debug("Audio stream closed")

    def _ensure_stream(self) -> Optional[str]:
        """Ensure the persistent stream is open on the current default device.

        Opens the stream on first use and reopens it only when the default input
        device actually changes (e.g. plugging in AirPods). The PortAudio device
        list is refreshed only while no stream is open — a safe window — so we
        pick up newly connected devices without disrupting a live stream.

        Must be called under self._lock.

        Returns:
            Name of the active device if it is Bluetooth, else None.
        """
        logger = get_logger()

        device_index, device_info = get_default_input_device(force_refresh=self.stream is None)
        device_name = device_info.get('name', '')
        logger.info(f"Current input device: {device_name} (index: {device_index})")

        if self.stream is not None and device_index == self._stream_device_index:
            logger.debug("Reusing existing stream")
        else:
            if self.stream is not None:
                logger.info(f"Input device changed to {device_name}; reopening stream")
            self._open_with_fallback(device_index, device_name)

        bluetooth = is_bluetooth_device(self.current_device_name)
        if bluetooth:
            self.activated_bluetooth_device = self.current_device_name
        elif self.debug:
            console.print(f"[dim]Input device: {self.current_device_name} (not Bluetooth)[/dim]")
        return self.current_device_name if bluetooth else None

    def _open_with_fallback(self, device_index: int | None, device_name: str) -> None:
        """Open the stream, retrying with the system default before giving up."""
        logger = get_logger()
        self._close_stream()  # no-op if nothing open; clears any stale reference
        try:
            self._open_stream(device_index, device_name)
            return
        except Exception as e:
            logger.warning(f"Stream open failed on {device_name}: {e}")

        # Retry once after letting CoreAudio settle, re-querying the device.
        time.sleep(0.3)
        device_index, device_info = get_default_input_device()
        device_name = device_info.get('name', 'Unknown')
        try:
            self._open_stream(device_index, device_name)
            return
        except Exception as e:
            logger.warning(f"Stream open retry failed on {device_name}: {e}")

        # Final fallback: let PortAudio pick the default device.
        logger.info("Falling back to system default device (device=None)")
        try:
            self._open_stream(None, "System Default")
        except Exception as e:
            self.stream = None
            raise RuntimeError(
                "Error starting stream: Internal PortAudio error [PaErrorCode -9986]\n"
                "This usually means macOS cannot access the microphone.\n"
                "Fix: System Settings → Privacy & Security → Microphone → enable your terminal app"
            ) from e

    def start(self) -> Optional[str]:
        """Start recording audio.

        The persistent stream stays running between recordings; this just opens it
        on first use (or reopens it on a device change) and flips the recording flag.

        Returns:
            Name of the active Bluetooth device (if any), for session tracking.
        """
        logger = get_logger()
        logger.info("=== RECORDING START ===")

        with self._lock:
            if self.recording:
                logger.error("Recording already in progress - cannot start new recording")
                raise RuntimeError("Recording already in progress")

            activated = self._ensure_stream()

            self.audio_data = []
            self._consecutive_silent_buffers = 0
            self._max_consecutive_silent = 0
            self._total_silent_buffers = 0
            self._total_buffers = 0
            self._status_errors = []
            self._silence_onset_buffer = None
            self.start_time = time.time()
            self.recording = True  # flip last so the callback only collects now

        logger.info(f"Recording started successfully on {self.current_device_name}")

        if self.debug:
            console.print(f"[green]Recording started at {self.sample_rate}Hz on {self.current_device_name}[/green]")

        return activated

    def stop(self) -> np.ndarray:
        """Stop recording and return the audio data.

        The stream is left running — only the recording flag is cleared, so the
        callback stops collecting. No stream stop/close or PortAudio reinit happens
        here, which is what previously hung after long sessions.

        Returns:
            Audio data as a numpy array
        """
        logger = get_logger()
        logger.info("=== RECORDING STOP ===")

        with self._lock:
            if not self.recording:
                logger.error("No recording in progress - cannot stop")
                raise RuntimeError("No recording in progress")
            # Flip the flag so the callback stops appending; the stream stays live.
            self.recording = False

        # Combine all audio chunks
        logger.debug(f"Processing {len(self.audio_data)} audio chunks")
        if not self.audio_data:
            logger.warning("No audio data captured")
            return np.array([], dtype=self.dtype)

        audio = np.concatenate(self.audio_data, axis=0)

        # Flatten to 1D if it's a multi-channel recording
        if audio.ndim > 1:
            logger.debug(f"Flattening audio from {audio.ndim}D to 1D")
            audio = audio.flatten()

        if self.start_time:
            duration = time.time() - self.start_time
            samples = len(audio)
            overall_rms = np.sqrt(np.mean(audio ** 2))
            logger.info(f"Recording stopped: {samples} samples, {duration:.2f}s, RMS={overall_rms:.4f}")

            if self.debug:
                console.print(f"[blue]Recording stopped: {samples} samples, {duration:.2f}s[/blue]")

        # Audio health diagnostics
        self._log_audio_diagnostics(audio)

        return audio

    def _log_audio_diagnostics(self, audio: np.ndarray) -> None:
        """Log diagnostics about audio quality, silence regions, and callback errors."""
        logger = get_logger()

        total = self._total_buffers
        silent = self._total_silent_buffers
        max_silent = self._max_consecutive_silent
        status_errors = self._status_errors

        if total == 0:
            return

        silent_pct = (silent / total) * 100
        is_bluetooth = is_bluetooth_device(self.current_device_name)

        logger.info(
            f"Audio diagnostics: {total} buffers, {silent} silent ({silent_pct:.1f}%), "
            f"max consecutive silent: {max_silent}, status errors: {len(status_errors)}"
        )

        # Detect trailing silence (audio cutout pattern)
        has_trailing_cutout = False
        if len(self.audio_data) > 10:
            # Check last 25% of buffers for silence
            tail_start = len(self.audio_data) * 3 // 4
            tail_silent = sum(
                1 for chunk in self.audio_data[tail_start:]
                if np.sqrt(np.mean(chunk ** 2)) < self._silence_threshold
            )
            tail_total = len(self.audio_data) - tail_start
            tail_silent_pct = (tail_silent / tail_total) * 100
            if tail_silent_pct > 80:
                has_trailing_cutout = True
                logger.warning(
                    f"AUDIO CUTOUT DETECTED: last 25% of recording is {tail_silent_pct:.0f}% silent "
                    f"(device: {self.current_device_name}, bluetooth: {is_bluetooth})"
                )

        if self.debug:
            console.print(f"[dim]Audio health: {total} buffers, {silent} silent ({silent_pct:.1f}%)[/dim]")
            console.print(f"[dim]Max consecutive silent buffers: {max_silent}[/dim]")

            if status_errors:
                console.print(f"[yellow]Callback status errors ({len(status_errors)}):[/yellow]")
                # Deduplicate and count
                from collections import Counter
                for err, count in Counter(status_errors).items():
                    console.print(f"[yellow]  {err} (x{count})[/yellow]")

            if has_trailing_cutout:
                console.print(
                    f"[red]⚠ Audio cutout detected: recording went silent in the last 25%. "
                    f"This is common with Bluetooth devices (AirPods) losing the HSP/HFP profile.[/red]"
                )
            elif silent_pct > 50:
                console.print(f"[yellow]⚠ High silence ratio ({silent_pct:.0f}%) — check microphone connection[/yellow]")

        # Reset counters
        self._consecutive_silent_buffers = 0
        self._max_consecutive_silent = 0
        self._total_silent_buffers = 0
        self._total_buffers = 0
        self._status_errors = []
        self._silence_onset_buffer = None

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.recording

    def close(self) -> None:
        """Close the persistent audio stream. Call once on shutdown."""
        with self._lock:
            self.recording = False
            self._close_stream()

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
            console.print(f"[dim]Saved recording to {filename}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Failed to save debug recording: {e}[/yellow]")
