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

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback for audio stream to capture audio data."""
        if status:
            logger = get_logger()
            logger.warning(f"Audio callback status: {status}")
            if self.debug:
                console.print(f"[yellow]Audio status: {status}[/yellow]")

        if self.recording:
            self.audio_data.append(indata.copy())

    def activate_bluetooth_device(self) -> Optional[str]:
        """Activate Bluetooth device by doing a brief test recording.

        This forces macOS to switch the Bluetooth profile from A2DP (audio output only)
        to HSP/HFP (bidirectional audio with microphone support).

        Returns:
            Name of the activated device, or None if no activation was needed/performed.
        """
        logger = get_logger()
        logger.debug("activate_bluetooth_device() called")

        # Get current default device.
        # Only force-refresh PortAudio if we previously had a Bluetooth device
        # (to detect if user switched away). Otherwise avoid disrupting CoreAudio.
        needs_refresh = self.activated_bluetooth_device is not None
        self.current_device_index, device_info = get_default_input_device(force_refresh=needs_refresh)
        self.current_device_name = device_info.get('name', '')
        device_name = self.current_device_name

        logger.info(f"Current input device: {device_name} (index: {self.current_device_index})")

        # Not a Bluetooth device - no activation needed
        if not is_bluetooth_device(device_name):
            logger.debug(f"Device is not Bluetooth: {device_name}")
            if self.debug:
                console.print(f"[dim]Input device: {device_name} (not Bluetooth)[/dim]")
            return None

        # Same Bluetooth device already activated - skip
        if device_name == self.activated_bluetooth_device:
            logger.debug(f"Bluetooth device already activated: {device_name}")
            if self.debug:
                console.print(f"[dim]Bluetooth device already activated: {device_name}[/dim]")
            return self.activated_bluetooth_device

        # New or different Bluetooth device - need to activate
        logger.info(f"Activating new Bluetooth device: {device_name}")
        if self.debug:
            console.print(f"[dim]Bluetooth device detected: {device_name}[/dim]")
            console.print("[dim]Activating Bluetooth microphone profile...[/dim]")

        try:
            logger.debug("Creating Bluetooth test stream")
            # Do a very brief recording to trigger the profile switch
            test_stream = sd.InputStream(
                device=self.current_device_index,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=1024,
            )
            logger.debug("Starting Bluetooth test stream")
            test_stream.start()
            logger.debug("Bluetooth test stream started, sleeping 0.3s")
            time.sleep(0.3)  # Brief delay to allow profile switch

            # Use timeouts to prevent hanging on Bluetooth issues
            logger.debug("Stopping Bluetooth test stream")
            _run_with_timeout(test_stream.stop, STREAM_TIMEOUT, "Bluetooth test stop")
            logger.debug("Closing Bluetooth test stream")
            _run_with_timeout(test_stream.close, STREAM_TIMEOUT, "Bluetooth test close")

            self.activated_bluetooth_device = device_name
            logger.info(f"Bluetooth device activated successfully: {device_name}")

            if self.debug:
                console.print("[dim]✓ Bluetooth microphone activated[/dim]")

            return device_name

        except Exception as e:
            logger.exception(f"Failed to activate Bluetooth device: {e}")
            if self.debug:
                console.print(f"[yellow]Warning: Failed to pre-activate Bluetooth: {e}[/yellow]")
            return None

    def _try_start_stream(self, device_index: int | None, device_name: str) -> bool:
        """Try to create and start an audio input stream.

        Args:
            device_index: PortAudio device index, or None for system default
            device_name: Device name for logging

        Returns:
            True if stream started successfully
        """
        logger = get_logger()
        logger.debug(f"Attempting stream start: device={device_index} ({device_name})")
        try:
            self.stream = sd.InputStream(
                device=device_index,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback,
            )
            self.stream.start()
            self.current_device_index = device_index
            self.current_device_name = device_name
            logger.info(f"Stream started on {device_name}")
            return True
        except Exception as e:
            logger.warning(f"Stream start failed on {device_name}: {e}")
            self.stream = None
            return False

    def start(self) -> Optional[str]:
        """Start recording audio.

        Returns:
            Name of the activated Bluetooth device (if any), for session tracking.
        """
        logger = get_logger()
        logger.info("=== RECORDING START ===")

        if self.recording:
            logger.error("Recording already in progress - cannot start new recording")
            raise RuntimeError("Recording already in progress")

        # Activate Bluetooth device if needed (checks if device changed)
        logger.debug("Activating Bluetooth device (if needed)")
        activated = self.activate_bluetooth_device()

        self.audio_data = []
        self.recording = True
        self.start_time = time.time()

        # Try to start the stream, with retry and fallback
        if not self._try_start_stream(self.current_device_index, self.current_device_name):
            # Retry after a brief delay to let CoreAudio settle
            logger.info("Retrying stream start after delay...")
            time.sleep(0.3)

            # Re-query devices in case indices changed
            self.current_device_index, device_info = get_default_input_device()
            self.current_device_name = device_info.get('name', 'Unknown')

            if not self._try_start_stream(self.current_device_index, self.current_device_name):
                # Final fallback: let PortAudio pick the default device
                logger.info("Falling back to system default device (device=None)")
                if not self._try_start_stream(None, "System Default"):
                    self.recording = False
                    raise RuntimeError(
                        "Error starting stream: Internal PortAudio error [PaErrorCode -9986]\n"
                        "This usually means macOS cannot access the microphone.\n"
                        "Fix: System Settings → Privacy & Security → Microphone → enable your terminal app"
                    )

        logger.info(f"Recording started successfully on {self.current_device_name}")

        if self.debug:
            console.print(f"[green]Recording started at {self.sample_rate}Hz on {self.current_device_name}[/green]")

        return activated

    def stop(self) -> np.ndarray:
        """Stop recording and return the audio data.

        Returns:
            Audio data as a numpy array
        """
        logger = get_logger()
        logger.info("=== RECORDING STOP ===")

        if not self.recording:
            logger.error("No recording in progress - cannot stop")
            raise RuntimeError("No recording in progress")

        self.recording = False
        logger.debug("Recording flag set to False")

        # Stop and close the stream with timeout to prevent hanging
        if self.stream:
            logger.debug("Stopping audio stream")
            stream = self.stream
            self.stream = None  # Clear reference first to prevent re-entry

            _run_with_timeout(stream.stop, STREAM_TIMEOUT, "stream stop")
            _run_with_timeout(stream.close, STREAM_TIMEOUT, "stream close")
            logger.debug("Audio stream stopped and closed")

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
            logger.info(f"Recording stopped: {samples} samples, {duration:.2f}s, RMS={np.sqrt(np.mean(audio**2)):.4f}")

            if self.debug:
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
            console.print(f"[dim]Saved recording to {filename.name}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Failed to save debug recording: {e}[/yellow]")
