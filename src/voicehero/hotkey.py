"""Hotkey listener for recording control."""

from typing import Callable

from pynput import keyboard

from .logger import get_logger


class HotkeyListener:
    """Listens for hotkey combinations to control recording."""

    def __init__(
        self,
        hotkeys: list[str],
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
    ):
        """Initialize the hotkey listener.

        Args:
            hotkeys: List of keys that must be held together (e.g., ['ctrl', 'cmd'])
            on_start: Callback when hotkey combination is pressed
            on_stop: Callback when hotkey combination is released
        """
        self.hotkeys = set(self._normalize_keys(hotkeys))
        self.on_start = on_start
        self.on_stop = on_stop
        self.pressed_keys: set[str] = set()
        self.was_active = False
        self.listener: keyboard.Listener | None = None

    def _normalize_keys(self, keys: list[str]) -> list[str]:
        """Normalize key names to pynput format."""
        normalized = []
        for key in keys:
            key_lower = key.lower()
            # Map common key names
            if key_lower in ("ctrl", "control"):
                normalized.append("ctrl")
            elif key_lower in ("cmd", "command", "meta", "super"):
                normalized.append("cmd")
            elif key_lower in ("alt", "option"):
                normalized.append("alt")
            elif key_lower == "shift":
                normalized.append("shift")
            else:
                normalized.append(key_lower)
        return normalized

    def _get_key_name(self, key) -> str:
        """Get normalized name for a key."""
        try:
            # Handle special keys
            if isinstance(key, keyboard.Key):
                if key == keyboard.Key.ctrl or key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                    return "ctrl"
                elif key == keyboard.Key.cmd or key == keyboard.Key.cmd_l or key == keyboard.Key.cmd_r:
                    return "cmd"
                elif key == keyboard.Key.alt or key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                    return "alt"
                elif key == keyboard.Key.shift or key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                    return "shift"
                else:
                    return key.name.lower()
            # Handle character keys
            elif hasattr(key, "char") and key.char:
                return key.char.lower()
            else:
                return str(key).lower()
        except AttributeError:
            return str(key).lower()

    def _on_press(self, key) -> None:
        """Handle key press events."""
        key_name = self._get_key_name(key)
        self.pressed_keys.add(key_name)

        # Check if hotkey combination is active
        is_active = self.hotkeys.issubset(self.pressed_keys)

        if is_active and not self.was_active:
            logger = get_logger()
            logger.info(f"Hotkey PRESSED: {sorted(self.pressed_keys)}")
            self.was_active = True
            try:
                self.on_start()
                logger.debug("on_start() callback completed")
            except Exception as e:
                logger.exception(f"Error in on_start() callback: {e}")

    def _on_release(self, key) -> None:
        """Handle key release events."""
        key_name = self._get_key_name(key)
        self.pressed_keys.discard(key_name)

        # Check if hotkey combination is no longer active
        is_active = self.hotkeys.issubset(self.pressed_keys)

        if not is_active and self.was_active:
            logger = get_logger()
            logger.info(f"Hotkey RELEASED: {sorted(self.pressed_keys)}")
            self.was_active = False
            try:
                self.on_stop()
                logger.debug("on_stop() callback completed")
            except Exception as e:
                logger.exception(f"Error in on_stop() callback: {e}")

    def start(self) -> None:
        """Start listening for hotkeys."""
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.start()

    def stop(self, timeout: float = 1.0) -> None:
        """Stop listening for hotkeys.

        Args:
            timeout: Maximum time to wait for listener thread to stop
        """
        if self.listener:
            self.listener.stop()
            # Wait for thread to finish with timeout to prevent hanging
            self.listener.join(timeout=timeout)
            self.listener = None
