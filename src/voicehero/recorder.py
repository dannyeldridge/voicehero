"""Audio recording functionality."""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from rich.console import Console

console = Console()


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


def get_default_input_device() -> tuple[int | None, dict]:
    """Get the current default input device, refreshing the device list.

    Returns:
        Tuple of (device_index, device_info_dict)
    """
    try:
        # Refresh device list to pick up any changes
        sd._terminate()
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
        if status and self.debug:
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
        # Get current default device (refreshes device list)
        self.current_device_index, device_info = get_default_input_device()
        self.current_device_name = device_info.get('name', '')
        device_name = self.current_device_name

        # Not a Bluetooth device - no activation needed
        if not is_bluetooth_device(device_name):
            if self.debug:
                console.print(f"[dim]Input device: {device_name} (not Bluetooth)[/dim]")
            return None

        # Same Bluetooth device already activated - skip
        if device_name == self.activated_bluetooth_device:
            if self.debug:
                console.print(f"[dim]Bluetooth device already activated: {device_name}[/dim]")
            return self.activated_bluetooth_device

        # New or different Bluetooth device - need to activate
        if self.debug:
            console.print(f"[dim]Bluetooth device detected: {device_name}[/dim]")
            console.print("[dim]Activating Bluetooth microphone profile...[/dim]")

        try:
            # Do a very brief recording to trigger the profile switch
            test_stream = sd.InputStream(
                device=self.current_device_index,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=1024,
            )
            test_stream.start()
            time.sleep(0.3)  # Brief delay to allow profile switch
            test_stream.stop()
            test_stream.close()

            self.activated_bluetooth_device = device_name

            if self.debug:
                console.print("[dim]✓ Bluetooth microphone activated[/dim]")

            return device_name

        except Exception as e:
            if self.debug:
                console.print(f"[yellow]Warning: Failed to pre-activate Bluetooth: {e}[/yellow]")
            return None

    def start(self) -> Optional[str]:
        """Start recording audio.

        Returns:
            Name of the activated Bluetooth device (if any), for session tracking.
        """
        if self.recording:
            raise RuntimeError("Recording already in progress")

        # Activate Bluetooth device if needed (checks if device changed)
        activated = self.activate_bluetooth_device()

        self.audio_data = []
        self.recording = True
        self.start_time = time.time()

        # Create and start the audio stream using the current default device
        self.stream = sd.InputStream(
            device=self.current_device_index,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._audio_callback,
        )
        self.stream.start()

        if self.debug:
            console.print(f"[green]Recording started at {self.sample_rate}Hz on {self.current_device_name}[/green]")

        return activated

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
            console.print(f"[dim]Saved recording to {filename.name}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Failed to save debug recording: {e}[/yellow]")
