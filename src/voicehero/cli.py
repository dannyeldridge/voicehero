"""Main CLI entry point for VoiceHero."""

import signal
import subprocess
import sys
import time

# Print immediately before heavy imports
print("Initializing VoiceHero...")

import numpy as np
import pyperclip
import typer
from rich.console import Console
from rich.panel import Panel

from .config import get_recordings_dir, load_config
from .config_cmd import config_command
from .hotkey import HotkeyListener
from .recorder import AudioRecorder, get_default_input_device, is_bluetooth_device
from .transcriber import AudioTranscriber

console = Console()


def check_macos() -> None:
    """Check if running on macOS."""
    if sys.platform != "darwin":
        console.print("\n[red]❌ ERROR: VoiceHero only supports macOS[/red]\n")
        console.print("VoiceHero requires macOS-specific features:")
        console.print("  - Global hotkey support via Accessibility APIs")
        console.print("  - Audio recording")
        console.print("  - AppleScript for auto-paste\n")
        raise typer.Exit(1)


def auto_paste(debug: bool = False) -> bool:
    """Paste clipboard contents using AppleScript.

    Args:
        debug: If True, show detailed error information

    Returns:
        True if paste succeeded, False otherwise
    """
    try:
        # Give the system a moment to ensure clipboard is ready
        time.sleep(0.3)

        # Use a more robust AppleScript that activates the frontmost app first
        script = '''
        tell application "System Events"
            keystroke "v" using command down
        end tell
        '''

        result = subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )

        # Additional delay to let the paste complete
        time.sleep(0.15)
        return True

    except Exception as e:
        console.print(f"[yellow]⚠️  Auto-paste failed[/yellow]")
        console.print("[dim]Text is still in clipboard - you can paste manually with Cmd+V[/dim]")
        console.print("[dim]Note: Terminal needs Accessibility permissions in System Settings[/dim]")

        if debug:
            console.print(f"[dim]Error details: {type(e).__name__}: {e}[/dim]")
            if hasattr(e, 'stderr') and e.stderr:
                console.print(f"[dim]stderr: {e.stderr}[/dim]")
            if hasattr(e, 'stdout') and e.stdout:
                console.print(f"[dim]stdout: {e.stdout}[/dim]")

        return False


def run(
    model: str = typer.Option(None, "--model", "-m", help="Whisper model size (tiny, base, small, medium, large)"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode with extra logging"),
) -> None:
    """Launch voice-to-text transcription."""
    check_macos()

    # Load or create config
    config = load_config()
    if not config:
        console.print("[yellow]Welcome to VoiceHero![/yellow]")
        console.print("Let's configure your preferences first.\n")

        from .config_cmd import interactive_config

        interactive_config()

        # Reload config
        config = load_config()
        if not config:
            console.print("[red]Configuration was not saved. Please run 'voicehero config' to set up.[/red]")
            raise typer.Exit(1)

    # Determine model to use
    model_to_use = model or config.model

    # Display header
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]VoiceHero - Voice-to-Text Transcriber[/bold cyan]")
    console.print("=" * 60)
    console.print(f"[cyan]Hotkey:[/cyan] {' + '.join(config.hotkey)}")
    console.print(f"[cyan]Model:[/cyan] {model_to_use}")
    console.print(f"[cyan]Auto-paste:[/cyan] {'Yes' if config.auto_paste else 'No'}")
    console.print("\n[bold]Instructions:[/bold]")
    console.print(f"  - Press and HOLD [cyan]{' + '.join(config.hotkey)}[/cyan] to record")
    console.print(f"  - Release to transcribe{' and paste' if config.auto_paste else ''}")
    console.print("  - Press [cyan]Ctrl+C[/cyan] to exit")
    console.print("\n" + "=" * 60 + "\n")

    # Initialize transcriber
    transcriber = AudioTranscriber(model_size=model_to_use)

    try:
        transcriber.initialize()
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise typer.Exit(1)

    # Show accessibility warning
    console.print(
        Panel(
            "[yellow]⚠️  ACCESSIBILITY PERMISSIONS REQUIRED[/yellow]\n\n"
            "VoiceHero uses global hotkeys and requires macOS\n"
            "Accessibility permissions to monitor keyboard events.\n\n"
            "To grant permissions:\n"
            "  1. Go to System Settings → Privacy & Security → Accessibility\n"
            "  2. Add your terminal app (Terminal, iTerm2, etc.)\n"
            "  3. Enable the checkbox",
            border_style="yellow",
        )
    )
    console.print()

    if debug:
        recordings_dir = get_recordings_dir()
        console.print(f"[dim]🐛 DEBUG MODE: Recordings will be saved to {recordings_dir}[/dim]")
        console.print(f"[dim]   Recordings will be automatically deleted when you exit.[/dim]\n")

    # Check for Bluetooth devices and show warning
    _, device_info = get_default_input_device()
    device_name = device_info.get('name', 'Unknown')

    if debug:
        console.print(f"[dim]📱 Input device: {device_name}[/dim]\n")

    if is_bluetooth_device(device_name):
        console.print(
            Panel(
                "[yellow]🎧 BLUETOOTH DEVICE DETECTED[/yellow]\n\n"
                f"Input: {device_name}\n\n"
                "Bluetooth headphones require profile switching to use the microphone.\n"
                "VoiceHero will automatically activate the microphone on first use.\n\n"
                "[dim]Note: Audio quality may be lower when the microphone is active.[/dim]",
                border_style="yellow",
            )
        )
        console.print()

    console.print(f"[green]🎤 Ready! Hold {' + '.join(config.hotkey)} to record...[/green]\n")

    # State management
    recorder: AudioRecorder | None = None
    is_transcribing = False
    activated_bluetooth_device: str | None = None  # Track activated Bluetooth device across recordings

    def on_start():
        nonlocal recorder, is_transcribing, activated_bluetooth_device

        if is_transcribing:
            return

        try:
            recorder = AudioRecorder(debug=debug, activated_bluetooth_device=activated_bluetooth_device)
            activated_bluetooth_device = recorder.start()
            console.print("[red]🔴 Recording...[/red]")
        except Exception as e:
            console.print(f"[red]Failed to start recording: {e}[/red]")

    def on_stop():
        nonlocal recorder, is_transcribing

        if not recorder or is_transcribing:
            return

        is_transcribing = True

        try:
            audio = recorder.stop()
            recorder = None

            if len(audio) == 0:
                console.print("[yellow]⚠️  No audio recorded[/yellow]\n")
                is_transcribing = False
                return

            console.print("[cyan]⏳ Transcribing...[/cyan]")

            if debug:
                rms = np.sqrt(np.mean(audio**2))
                duration = len(audio) / 16000
                console.print(f"[dim]Debug: samples={len(audio)}, duration={duration:.2f}s, RMS={rms:.4f}[/dim]")

            start_time = time.time()
            text = transcriber.transcribe(audio)
            elapsed = time.time() - start_time

            if text:
                console.print(f'\n[green]"{text}"[/green]')
                console.print(f"[dim](transcribed in {elapsed:.1f}s)[/dim]")

                # Copy to clipboard
                pyperclip.copy(text)

                # Auto-paste if configured
                if config.auto_paste:
                    if auto_paste(debug=debug):
                        console.print("[green]✓ Pasted![/green]\n")
                    else:
                        console.print()  # Add newline after error messages
                else:
                    console.print("[green]✓ Copied to clipboard[/green]\n")

                # Save debug recording
                if debug:
                    recorder_temp = AudioRecorder(debug=debug)
                    recorder_temp.save_debug_recording(audio, get_recordings_dir())
            else:
                console.print("[yellow]⚠️  No speech detected[/yellow]\n")

        except Exception as e:
            console.print(f"[red]Transcription error: {e}[/red]\n")
        finally:
            is_transcribing = False

    # Start hotkey listener
    listener = HotkeyListener(config.hotkey, on_start=on_start, on_stop=on_stop)
    listener.start()

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        console.print("\n\n[cyan]Goodbye![/cyan]")

        # Clean up debug recordings
        if debug:
            recordings_dir = get_recordings_dir()
            import shutil
            if recordings_dir.exists():
                try:
                    shutil.rmtree(recordings_dir)
                    console.print(f"[dim]Cleaned up debug recordings from {recordings_dir}[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Failed to clean up recordings: {e}[/yellow]")

        console.print()
        listener.stop()
        transcriber.dispose()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Keep process alive
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        signal_handler(None, None)


def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    reset: bool = typer.Option(False, "--reset", help="Reset to default configuration"),
) -> None:
    """Configure VoiceHero settings."""
    config_command(show=show, reset=reset)


# Create the app and register commands
app = typer.Typer(
    name="voicehero",
    help="Voice-to-text transcription using OpenAI's Whisper model",
    invoke_without_command=True,
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    model: str = typer.Option(None, "--model", "-m", help="Whisper model size (tiny, base, small, medium, large)"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode with extra logging"),
) -> None:
    """Voice-to-text transcription using OpenAI's Whisper model."""
    # If a subcommand was invoked, don't run the main function
    if ctx.invoked_subcommand is not None:
        return

    # Otherwise run the transcription
    run(model=model, debug=debug)


app.command(name="config")(config)


if __name__ == "__main__":
    app()
