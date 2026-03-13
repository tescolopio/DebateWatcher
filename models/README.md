# DebateWatcher Local Model Weights

This directory stores GGUF/GGML model weight files for local inference.

## Recommended Models

| Model | Size | Use Case |
|---|---|---|
| `base.en` | ~150 MB | Transcription (CPU) |
| `small.en` | ~460 MB | Transcription (higher accuracy) |
| `mistral:7b` (via Ollama) | ~4 GB | Fallacy detection |
| `llama3:8b` (via Ollama) | ~5 GB | Fallacy detection (recommended) |

## Whisper Models

Download Whisper models using faster-whisper (auto-downloads on first use):

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('base.en')"
```

## Ollama Models

Pull models via the Ollama CLI:

```bash
ollama pull llama3:8b
ollama pull mistral:7b
```

> **Note:** Model files are excluded from version control via `.gitignore`.
> Add `*.bin`, `*.gguf`, and `*.ggml` to your `.gitignore` if they are not already present.
