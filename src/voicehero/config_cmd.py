"""Configuration command for VoiceHero."""

import sys

import typer
from pynput import keyboard
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .config import VoiceHeroConfig, load_config, save_config

console = Console()


def show_config() -> None:
    """Display current configuration."""
    config = load_config()

    if not config:
        console.print("[yellow]No configuration found. Run 'voicehero config' to set up.[/yellow]")
        return

    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Hotkey", " + ".join(config.hotkey))
    table.add_row("Model", config.model)
    table.add_row("Auto-paste", "Yes" if config.auto_paste else "No")

    console.print(table)


def reset_config() -> None:
    """Reset configuration to defaults."""
    config = VoiceHeroConfig()
    save_config(config)

    console.print("\n[green]✓ Configuration reset to defaults[/green]")
    show_config()


def record_hotkey() -> list[str]:
    """Record a hotkey combination from the user."""
    console.print("\n[cyan]Press and HOLD your desired key combination, then press ENTER...[/cyan]")
    console.print("[dim](Press ENTER without holding keys to keep current hotkey)[/dim]\n")

    pressed_keys: set[str] = set()
    result: list[str] = []

    def on_press(key):
        try:
            if isinstance(key, keyboard.Key):
                if key == keyboard.Key.ctrl or key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                    pressed_keys.add("ctrl")
                elif key == keyboard.Key.cmd or key == keyboard.Key.cmd_l or key == keyboard.Key.cmd_r:
                    pressed_keys.add("cmd")
                elif key == keyboard.Key.alt or key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                    pressed_keys.add("alt")
                elif key == keyboard.Key.shift or key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                    pressed_keys.add("shift")
                elif key == keyboard.Key.enter:
                    # Capture the currently held keys
                    result.extend(sorted(pressed_keys))
                    return False  # Stop listener
            elif hasattr(key, "char") and key.char:
                pressed_keys.add(key.char.lower())
        except AttributeError:
            pass

    def on_release(key):
        try:
            if isinstance(key, keyboard.Key):
                if key == keyboard.Key.ctrl or key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                    pressed_keys.discard("ctrl")
                elif key == keyboard.Key.cmd or key == keyboard.Key.cmd_l or key == keyboard.Key.cmd_r:
                    pressed_keys.discard("cmd")
                elif key == keyboard.Key.alt or key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                    pressed_keys.discard("alt")
                elif key == keyboard.Key.shift or key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                    pressed_keys.discard("shift")
            elif hasattr(key, "char") and key.char:
                pressed_keys.discard(key.char.lower())
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    return result if result else ["ctrl", "cmd"]  # Default fallback


def interactive_config() -> None:
    """Run interactive configuration wizard."""
    console.print("\n[bold cyan]VoiceHero Configuration[/bold cyan]")
    console.print("=" * 40 + "\n")

    existing_config = load_config() or VoiceHeroConfig()

    # 1. Auto-paste preference
    console.print("When transcription completes, the text can be automatically")
    console.print("pasted at your cursor position, or just copied to clipboard.\n")

    auto_paste = Confirm.ask(
        "Auto-paste after transcribing?",
        default=existing_config.auto_paste,
    )

    # 2. Model selection
    console.print("\n[bold]Whisper model sizes:[/bold]")
    console.print("  [dim]tiny[/dim]      (~75MB)   - Fastest, least accurate")
    console.print("  [cyan]base[/cyan]      (~150MB)  - Fast, basic accuracy (default)")
    console.print("  [dim]small[/dim]     (~500MB)  - Good balance")
    console.print("  [dim]medium[/dim]    (~1.5GB)  - Better accuracy, slower")
    console.print("  [dim]large[/dim]     (~3GB)    - Best accuracy, slowest\n")

    valid_models = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
    model = Prompt.ask(
        "Model",
        default=existing_config.model,
        choices=list(valid_models),
    )

    # 3. Hotkey configuration
    console.print("\n[bold]Hotkey configuration[/bold]")
    console.print("This is the key combination you'll hold to record.\n")

    change_hotkey = Confirm.ask(
        f"Change hotkey? (current: {' + '.join(existing_config.hotkey)})",
        default=False,
    )

    hotkey = existing_config.hotkey
    if change_hotkey:
        recorded = record_hotkey()
        if recorded:
            hotkey = recorded

    # Save configuration
    new_config = VoiceHeroConfig(
        auto_paste=auto_paste,
        model=model,
        hotkey=hotkey,
    )

    save_config(new_config)

    console.print("\n" + "=" * 40)
    console.print("[bold green]✓ Configuration saved![/bold green]")
    console.print("=" * 40)
    console.print(f"  Auto-paste: [cyan]{'Yes' if new_config.auto_paste else 'No'}[/cyan]")
    console.print(f"  Hotkey: [cyan]{' + '.join(new_config.hotkey)}[/cyan]")
    console.print(f"  Model: [cyan]{new_config.model}[/cyan]")
    console.print("\n[dim]Run 'voicehero' to start transcribing.[/dim]\n")


def config_command(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    reset: bool = typer.Option(False, "--reset", help="Reset to default configuration"),
) -> None:
    """Configure VoiceHero settings."""
    if show:
        show_config()
    elif reset:
        reset_config()
    else:
        interactive_config()
