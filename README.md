# VoiceHero

Voice-to-text transcription CLI

## Features

- **Live transcription**: Press-and-hold hotkey to record, release to transcribe
- **File conversion**: Transcribe existing audio files (wav, mp3, m4a, ogg, flac, etc.)
- **Fast transcription**: Powered by faster-whisper
- **Configurable models**: Choose from tiny to large Whisper models (quality vs. speed)
- **Smart clipboard**: Automatic clipboard copy and optional auto-paste
- **Customizable hotkeys**: Configure any key combination for recording
- **Real-time feedback**: Live status updates and progress bars
- **Statistics tracking**: Session and lifetime word count with time saved calculations
- **Debug mode**: Optional audio recording and detailed logging for troubleshooting

## Requirements

- Python >= 3.10
- macOS (for global hotkey support and auto-paste)
- **Accessibility permissions** for your terminal app (required for global hotkeys)

### Accessibility Permissions Setup

VoiceHero uses global hotkeys which require macOS Accessibility permissions.

**To grant permissions:**
1. Go to **System Settings → Privacy & Security → Accessibility**
2. Add your terminal app (Terminal.app, iTerm2, etc.)
3. Enable the checkbox

This prevents permission prompts during use.

## Installation

### Quick Install (Recommended)

Install directly from GitHub using UV:

```bash
# Install UV if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install VoiceHero globally
uv tool install git+https://github.com/dannyeldridge/voicehero.git

# Or install a specific version
uv tool install git+https://github.com/dannyeldridge/voicehero.git@v0.1.0
```

### Development Installation

For local development:

```bash
# Clone the repository
git clone https://github.com/yourusername/voicehero.git
cd voicehero

# Install dependencies (creates virtual environment automatically)
uv sync

# Run locally
uv run voicehero
```

## Usage

VoiceHero has two main modes: **live transcription** and **file conversion**.

### Live Transcription

```bash
voicehero
```

On first run, you'll be guided through configuration (hotkey, model, auto-paste preference).

Then the transcriber starts listening in the background:
1. Press and HOLD your configured hotkey to record
2. Speak into your microphone
3. Release the hotkey to transcribe
4. The transcribed text is automatically copied to your clipboard (and optionally pasted)

### File Conversion

Transcribe existing audio files:

```bash
voicehero convert recording.mp3
voicehero convert meeting.wav --level 5 --output transcript.txt
```

Perfect for transcribing:
- Voice memos and recordings
- Meeting recordings
- Podcast episodes
- Video audio tracks
- Any audio in wav, mp3, m4a, ogg, flac, or other formats

### Commands

#### `voicehero`

Launch the live voice transcriber with hotkey activation.

```bash
voicehero                          # Start with saved config
voicehero --model base             # Override model size
voicehero --debug                  # Enable debug mode with detailed logging
voicehero --save-recordings        # Save audio recordings to ~/.voicehero/recordings/
voicehero --debug --save-recordings # Debug mode with recordings preserved
```

**Options:**
- `--model`, `-m` - Whisper model size (tiny, base, small, medium, large)
- `--debug`, `-d` - Enable debug mode with extra logging and audio metrics
- `--save-recordings` - Save raw audio recordings to disk for review or debugging

**Debug mode** (`--debug` flag):
- Saves detailed logs to `~/.voicehero/recordings/voicehero-<timestamp>.log`
- Records all audio input to WAV files for analysis
- Shows additional console output (audio levels, timing, device info)
- Logs all events: hotkey presses, recording state, transcription progress
- Automatically cleans up debug files on exit
- Useful for diagnosing issues like hanging or audio problems

#### `voicehero convert`

Transcribe an existing audio file to text.

```bash
voicehero convert audio.mp3              # Transcribe with default level 3 (small model)
voicehero convert audio.wav --level 5    # Use level 5 (large model, best accuracy)
voicehero convert audio.m4a -l 1         # Use level 1 (tiny model, fastest)
voicehero convert audio.mp3 -o text.txt  # Save output to a file
```

**Options:**
- `--level`, `-l` - Model accuracy level 1-5 (default: 3)
  - Level 1: tiny (~75MB) - Fastest, basic accuracy
  - Level 2: base (~150MB) - Fast, good accuracy
  - Level 3: small (~500MB) - Balanced (default)
  - Level 4: medium (~1.5GB) - Better accuracy, slower
  - Level 5: large (~3GB) - Best accuracy, slowest
- `--output`, `-o` - Write transcription to a file

**Supported formats:** wav, mp3, m4a, ogg, flac, webm, mp4, aac, wma (any ffmpeg-compatible format)

**Output:**
- Prints transcribed text to terminal
- Copies to clipboard automatically
- Optionally writes to file with `--output`
- Shows word count and elapsed time

#### `voicehero config`

Configure voice transcription settings.

```bash
voicehero config         # Interactive configuration wizard
voicehero config --show  # Display current settings
voicehero config --reset # Reset to defaults
```

## Configuration

Settings are stored in \`~/.voicehero/config.json\`:

| Setting | Description | Default |
|---------|-------------|---------|
| \`hotkey\` | Key combination to hold for recording | \`["ctrl", "cmd"]\` |
| \`model\` | Whisper model size | \`base\` |
| \`auto_paste\` | Auto-paste after transcription | \`true\` |

### Model Sizes

| Model | Size | Speed | Accuracy | Level (convert) |
|-------|------|-------|----------|-----------------|
| tiny | ~75MB | Fastest | Basic | 1 |
| base | ~150MB | Fast | Good | 2 |
| small | ~500MB | Medium | Better | 3 |
| medium | ~1.5GB | Slower | High | 4 |
| large | ~3GB | Slowest | Best | 5 |
| large-v2 | ~3GB | Slowest | Best (v2) | - |
| large-v3 | ~3GB | Slowest | Best (v3) | - |

**Note:** Live transcription defaults to `base` for speed, while file conversion defaults to `small` (level 3) for better accuracy since there's no real-time constraint.

## Examples

### Live Transcription Workflow
```bash
# First time setup
voicehero config

# Start transcribing
voicehero

# Use with different model for better quality
voicehero --model large

# Debug microphone issues
voicehero --debug --save-recordings
```

### File Conversion Workflow
```bash
# Quick transcription with defaults
voicehero convert meeting.mp3

# High-accuracy transcription with output file
voicehero convert interview.wav --level 5 --output transcript.txt

# Fast transcription of multiple files (bash loop)
for file in recordings/*.mp3; do
  voicehero convert "$file" -l 1 -o "transcripts/$(basename "$file" .mp3).txt"
done
```

## Troubleshooting

### Microphone not working
1. Check System Settings → Privacy & Security → Microphone
2. Enable permissions for your terminal app
3. Restart your terminal

### Auto-paste not working
1. Check System Settings → Privacy & Security → Accessibility
2. Enable permissions for your terminal app
3. Try running `voicehero config` and disable auto-paste if issues persist

### Bluetooth headset microphone
VoiceHero automatically detects and activates Bluetooth microphones. The first recording may take a moment while the profile switches from high-quality audio (A2DP) to microphone mode (HSP/HFP).

### No speech detected
- Ensure you're speaking clearly during recording
- Try a different model size (smaller models may miss quiet audio)
- Check microphone levels in System Settings → Sound

## Development

```bash
# Install dependencies with UV
uv sync

# Run locally
uv run voicehero
