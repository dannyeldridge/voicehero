"""Main CLI entry point for VoiceHero."""

import logging
import os
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

from .config import get_recordings_dir, load_config, load_stats, save_stats
from .config_cmd import config_command
from .convert_cmd import convert
from .hotkey import HotkeyListener
from .logger import init_logger, get_logger
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
    model: str,
    debug: bool,
    save_recordings: bool,
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

    # Initialize logger
    if debug:
        log_dir = get_recordings_dir()
        init_logger(debug=True, log_dir=log_dir)
        logger = get_logger()
        logger.info(f"VoiceHero starting in DEBUG mode")
        logger.info(f"Model: {model_to_use}")
        logger.info(f"Hotkey: {config.hotkey}")
        logger.info(f"Auto-paste: {config.auto_paste}")
    else:
        init_logger(debug=False)

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
        console.print(f"[dim]🐛 DEBUG MODE: Recordings and logs will be saved to {recordings_dir}[/dim]")
        if save_recordings:
            console.print(f"[dim]   Audio recordings will be saved.[/dim]")
        console.print(f"[dim]   Debug files will be automatically deleted when you exit.[/dim]")
        # Show the actual log file path
        logger = get_logger()
        for handler in logger.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                console.print(f"[dim]   Log file: {handler.baseFilename}[/dim]")
        console.print()

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

    # Load statistics
    stats = load_stats()

    # Display total stats if any exist
    if stats.total_words > 0:
        total_minutes_saved = stats.total_words / 40
        console.print(
            f"[dim]📊 Total lifetime: {stats.total_transcriptions} transcriptions • "
            f"{stats.total_words} words • ~{total_minutes_saved:.1f} min saved[/dim]\n"
        )

    console.print(f"[green]🎤 Ready! Hold {' + '.join(config.hotkey)} to record...[/green]\n")

    # State management. The recorder is persistent: created once and reused for
    # every recording so its audio stream stays open for the whole session.
    recorder = AudioRecorder(debug=debug)
    is_transcribing = False
    session_word_count = 0  # Track total words transcribed in this session

    def on_start():
        nonlocal is_transcribing

        if debug:
            logger = get_logger()
            logger.info(">>> on_start() called")
            logger.debug(f"State: is_transcribing={is_transcribing}, recording={recorder.is_recording()}")

        if is_transcribing or recorder.is_recording():
            if debug:
                logger.debug("Skipping on_start - transcription in progress or already recording")
            return

        try:
            recorder.start()
            console.print("[red]🔴 Recording...[/red]")
        except Exception as e:
            if debug:
                logger.exception(f"Failed to start recording: {e}")
            error_str = str(e)
            if "9986" in error_str or "Audio Hardware" in error_str or "not Running" in error_str:
                terminal = os.environ.get("TERM_PROGRAM", "your terminal app")
                console.print(
                    f"[red]Microphone access denied.[/red]\n"
                    f"[yellow]Grant microphone permission to {terminal}:[/yellow]\n"
                    f"  System Settings → Privacy & Security → Microphone → enable {terminal}\n"
                    f"  Then restart {terminal}."
                )
            else:
                console.print(f"[red]Failed to start recording: {e}[/red]")

    def on_stop():
        nonlocal is_transcribing, session_word_count

        if debug:
            logger = get_logger()
            logger.info(">>> on_stop() called")
            logger.debug(f"State: recording={recorder.is_recording()}, is_transcribing={is_transcribing}")

        if is_transcribing or not recorder.is_recording():
            if debug:
                logger.debug(f"Skipping on_stop - recording={recorder.is_recording()}, is_transcribing={is_transcribing}")
            return

        is_transcribing = True
        if debug:
            logger.debug("Set is_transcribing=True")

        try:
            audio = recorder.stop()

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

                # Count words and update session total
                word_count = len(text.split())
                session_word_count += word_count

                # Update and save cumulative stats
                stats.total_words += word_count
                stats.total_transcriptions += 1
                save_stats(stats)

                if debug:
                    logger.info(f"Transcription successful: {word_count} words, session total: {session_word_count}")

                # Calculate time saved (40 WPM average typing speed)
                session_minutes_saved = session_word_count / 40
                total_minutes_saved = stats.total_words / 40

                # Copy to clipboard
                pyperclip.copy(text)
                if debug:
                    logger.debug("Text copied to clipboard")

                # Auto-paste if configured
                if config.auto_paste:
                    if debug:
                        logger.debug("Attempting auto-paste")
                    if auto_paste(debug=debug):
                        console.print("[green]✓ Pasted![/green]")
                        if debug:
                            logger.debug("Auto-paste successful")
                    else:
                        console.print()  # Add newline after error messages
                        if debug:
                            logger.warning("Auto-paste failed")
                else:
                    console.print("[green]✓ Copied to clipboard[/green]")

                # Display session and total stats
                console.print(
                    f"[dim](Session: {session_word_count} words • ~{session_minutes_saved:.1f} min saved | "
                    f"Total: {stats.total_words} words • ~{total_minutes_saved:.1f} min saved)[/dim]\n"
                )

                # Save recording if requested
                if save_recordings:
                    recorder.save_debug_recording(audio, get_recordings_dir())
            else:
                console.print("[yellow]⚠️  No speech detected[/yellow]\n")
                if debug:
                    logger.warning("No speech detected in transcription")

        except Exception as e:
            console.print(f"[red]Transcription error: {e}[/red]\n")
            if debug:
                logger.exception(f"Transcription error: {e}")
        finally:
            is_transcribing = False
            if debug:
                logger.debug("Set is_transcribing=False")
                logger.info(">>> on_stop() completed")

    # Start hotkey listener
    listener = HotkeyListener(config.hotkey, on_start=on_start, on_stop=on_stop)
    listener.start()

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        if debug:
            logger = get_logger()
            logger.info("=== SHUTDOWN INITIATED ===")

        console.print("\n\n[cyan]Goodbye![/cyan]")

        # Close the persistent audio stream to release the microphone.
        if debug:
            logger.debug("Closing audio stream")
        try:
            recorder.close()
        except Exception as e:
            if debug:
                logger.error(f"Error closing recorder during shutdown: {e}")
            pass  # Ignore errors during shutdown

        # Stop hotkey listener with timeout
        if debug:
            logger.debug("Stopping hotkey listener")
        listener.stop()

        # Clean up debug files after stopping recorder (unless recordings were saved)
        if debug and not save_recordings:
            logger.info("Cleaning up debug files")
            recordings_dir = get_recordings_dir()
            import shutil
            if recordings_dir.exists():
                try:
                    # Use ignore_errors to prevent hanging on locked files
                    shutil.rmtree(recordings_dir, ignore_errors=True)
                    console.print(f"[dim]Cleaned up debug files from {recordings_dir}[/dim]")
                    logger.debug("Debug files cleaned up successfully")
                except Exception as e:
                    console.print(f"[yellow]Failed to clean up recordings: {e}[/yellow]")
                    logger.error(f"Failed to clean up debug files: {e}")

        console.print()
        if debug:
            logger.debug("Disposing transcriber")
        transcriber.dispose()

        if debug:
            logger.info("=== SHUTDOWN COMPLETE ===")
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
    save_recordings: bool = typer.Option(False, "--save-recordings", help="Save audio recordings to disk"),
) -> None:
    """Voice-to-text transcription using OpenAI's Whisper model."""
    # If a subcommand was invoked, don't run the main function
    if ctx.invoked_subcommand is not None:
        return

    # Otherwise run the transcription
    run(model=model, debug=debug, save_recordings=save_recordings)


app.command(name="config")(config)
app.command(name="convert")(convert)


if __name__ == "__main__":
    app()
