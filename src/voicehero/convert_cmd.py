"""Convert (transcribe) an existing audio file to text."""

import time
from pathlib import Path
from typing import Annotated

import pyperclip
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from .logger import init_logger
from .transcriber import AudioTranscriber

console = Console()

LEVEL_TO_MODEL: dict[int, str] = {
    1: "tiny",
    2: "base",
    3: "small",
    4: "medium",
    5: "large",
}

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4", ".aac", ".wma"}


def convert(
    file: Annotated[Path, typer.Argument(help="Path to the audio file to transcribe")],
    level: Annotated[int, typer.Option("--level", "-l", min=1, max=5, help="Model accuracy level 1-5 (1=tiny/fast, 5=large/accurate)")] = 3,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Write transcription to a file")] = None,
) -> None:
    """Transcribe an audio file to text.

    Supports wav, mp3, m4a, ogg, flac, and other ffmpeg-compatible formats.
    """
    # Validate file
    if not file.exists():
        console.print(f"[red]Error: File not found: {file}[/red]")
        raise typer.Exit(1)

    suffix = file.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        console.print(f"[red]Error: Unsupported file type '{suffix}'[/red]")
        console.print(f"[dim]Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}[/dim]")
        raise typer.Exit(1)

    init_logger(debug=False)

    model_name = LEVEL_TO_MODEL[level]
    console.print(f"[cyan]File:[/cyan]  {file}")
    console.print(f"[cyan]Model:[/cyan] {model_name} (level {level})\n")

    # Initialize transcriber
    transcriber = AudioTranscriber(model_size=model_name)
    try:
        transcriber.initialize()
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise typer.Exit(1)

    # Transcribe with progress bar
    start_time = time.time()

    with Progress(
        TextColumn("[cyan]Transcribing..."),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("transcribe", total=100)

        def on_progress(current: float, total: float) -> None:
            if total > 0:
                pct = min(current / total * 100, 100)
                progress.update(task_id, completed=pct)

        text = transcriber.transcribe_file_with_progress(
            file,
            progress_callback=on_progress,
        )
        progress.update(task_id, completed=100)

    elapsed = time.time() - start_time
    transcriber.dispose()

    if not text:
        console.print("\n[yellow]No speech detected in the audio file.[/yellow]")
        raise typer.Exit(0)

    # Display result
    word_count = len(text.split())
    console.print(f'\n[green]"{text}"[/green]')
    console.print(f"\n[dim]{word_count} words | transcribed in {elapsed:.1f}s[/dim]")

    # Copy to clipboard
    pyperclip.copy(text)
    console.print("[green]Copied to clipboard[/green]")

    # Write to file if requested
    if output is not None:
        output.write_text(text, encoding="utf-8")
        console.print(f"[green]Written to {output}[/green]")
