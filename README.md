# VoiceHero

Voice-to-text transcription CLI

## Features

- Press-and-hold hotkey to record, release to transcribe
- Fast transcription with faster-whisper
- Configurable Whisper model sizes (tiny to large)
- Automatic clipboard copy and optional auto-paste
- Customizable hotkey combinations
- Real-time transcription feedback

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

### Start the transcriber

```bash
voicehero
```

On first run, you'll be guided through configuration (hotkey, model, auto-paste preference).

Then the transcriber starts listening in the background:
1. Press and HOLD your configured hotkey to record
2. Speak into your microphone
3. Release the hotkey to transcribe
4. The transcribed text is automatically copied to your clipboard (and optionally pasted)

### Commands

#### \`voicehero\`

Launch the voice transcriber.

```bash
voicehero              # Start with saved config
voicehero --model base # Override model size
voicehero --debug      # Show debug output (audio levels, timing)
```

#### \`voicehero config\`

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

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | ~75MB | Fastest | Basic |
| base | ~150MB | Fast | Good |
| small | ~500MB | Medium | Better |
| medium | ~1.5GB | Slower | High |
| large | ~3GB | Slowest | Best |
| large-v2 | ~3GB | Slowest | Best (v2) |
| large-v3 | ~3GB | Slowest | Best (v3) |

## Development

```bash
# Install dependencies with UV
uv sync

# Run locally
uv run voicehero
