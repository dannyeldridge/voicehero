"""Tests for AudioRecorder's persistent-stream behavior.

The only thing mocked is `sounddevice` (the hardware boundary). Everything
else exercises the real recorder logic.

These tests pin down the architectural fix for the PortAudio stop/close hangs:
the recorder must open ONE InputStream and keep it running for the process
lifetime, never tearing PortAudio down during normal start/stop cycles.
"""

import numpy as np
import pytest

from voicehero import recorder as recorder_module
from voicehero.logger import init_logger
from voicehero.recorder import AudioRecorder


@pytest.fixture(autouse=True)
def _logger():
    init_logger(debug=False)


class FakeStream:
    """Stand-in for sounddevice.InputStream that records lifecycle calls."""

    def __init__(self, registry, **kwargs):
        self.kwargs = kwargs
        self.callback = kwargs.get("callback")
        self.started = False
        self.stopped = False
        self.closed = False
        registry.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def feed(self, indata):
        """Simulate PortAudio delivering one buffer to the callback."""
        self.callback(indata, len(indata), None, None)


class FakeDefault:
    def __init__(self):
        self.device = [0, 1]


class FakeSd:
    """Minimal fake of the sounddevice module surface the recorder uses."""

    def __init__(self):
        self.instances: list[FakeStream] = []
        self.terminate_calls = 0
        self.initialize_calls = 0
        self.device_name = "MacBook Pro Microphone"
        self.device_index = 0
        self.default = FakeDefault()

    def InputStream(self, **kwargs):
        return FakeStream(self.instances, **kwargs)

    def query_devices(self, kind=None):
        return {"name": self.device_name}

    def _terminate(self):
        self.terminate_calls += 1

    def _initialize(self):
        self.initialize_calls += 1

    def set_device(self, index, name):
        self.device_index = index
        self.device_name = name
        self.default.device = [index, index + 1]


@pytest.fixture
def fake_sd(monkeypatch):
    fake = FakeSd()
    monkeypatch.setattr(recorder_module, "sd", fake)
    # Avoid real sleeps in retry/settle paths.
    monkeypatch.setattr(recorder_module.time, "sleep", lambda *_: None)
    return fake


def _buf(value=1.0, frames=1024):
    return np.full((frames, 1), value, dtype=np.float32)


def test_stream_opened_once_across_multiple_recordings(fake_sd):
    rec = AudioRecorder()

    for _ in range(3):
        rec.start()
        rec.stop()

    assert len(fake_sd.instances) == 1, "stream must be persistent, not per-recording"
    assert fake_sd.instances[0].started is True


def test_stop_does_not_stop_or_close_the_stream(fake_sd):
    rec = AudioRecorder()
    rec.start()
    rec.stop()

    stream = fake_sd.instances[0]
    assert stream.stopped is False
    assert stream.closed is False


def test_stop_never_reinitializes_portaudio(fake_sd):
    rec = AudioRecorder()
    rec.start()
    terminate_before = fake_sd.terminate_calls
    rec.stop()
    assert fake_sd.terminate_calls == terminate_before


def test_callback_collects_audio_only_while_recording(fake_sd):
    rec = AudioRecorder()
    rec.start()
    stream = fake_sd.instances[0]
    stream.feed(_buf())
    stream.feed(_buf())
    audio = rec.stop()

    # Buffer delivered after stop must be ignored.
    stream.feed(_buf())

    assert len(audio) == 2048


def test_each_recording_starts_with_a_fresh_buffer(fake_sd):
    rec = AudioRecorder()
    rec.start()
    fake_sd.instances[0].feed(_buf())
    rec.stop()

    rec.start()
    audio2 = rec.stop()
    assert len(audio2) == 0


def test_device_change_reopens_the_stream(fake_sd):
    rec = AudioRecorder()
    rec.start()
    rec.stop()

    fake_sd.set_device(2, "AirPods Pro")
    rec.start()
    rec.stop()

    assert len(fake_sd.instances) == 2, "device change should open a new stream"
    assert fake_sd.instances[0].closed is True, "old stream should be closed on switch"


def test_close_closes_the_persistent_stream(fake_sd):
    rec = AudioRecorder()
    rec.start()
    rec.stop()
    rec.close()

    assert fake_sd.instances[0].closed is True
