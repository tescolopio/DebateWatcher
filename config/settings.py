"""
config/settings.py – Centralised configuration for DebateWatcher.

All values are read from environment variables (or a .env file) and
validated with Pydantic-Settings, giving type-checked, documented
access to every tunable parameter from a single import.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OBSSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OBS_", env_file=".env", extra="ignore")

    host: str = Field("localhost", description="OBS WebSocket host")
    port: int = Field(4455, description="OBS WebSocket port")
    password: str = Field("", description="OBS WebSocket password")


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_", env_file=".env", extra="ignore")

    host: str = Field("http://localhost:11434", description="Ollama API base URL")
    model: str = Field("llama3:8b", description="LLM model identifier")
    analysis_temperature: float = Field(
        0.2, ge=0.0, le=2.0,
        description="LLM temperature for all analysis tasks (fallacy + contradiction)"
    )
    timeout: int = Field(30, ge=1, description="Request timeout in seconds")


class WhisperSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WHISPER_", env_file=".env", extra="ignore")

    backend: str = Field("faster-whisper", description="faster-whisper or whisper-cpp")
    model: str = Field("base.en", description="Whisper model size")
    device: str = Field("auto", description="cpu | cuda | auto")
    compute_type: str = Field("float16", description="float32 | float16 | int8")


class AudioSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUDIO_", env_file=".env", extra="ignore")

    device: str = Field("", description="Audio input device name (empty = system default)")
    sample_rate: int = Field(16000, description="Sample rate expected by Whisper")
    chunk_seconds: float = Field(4.0, ge=1.0, le=30.0, description="Sliding window size")
    chunk_overlap: float = Field(0.5, ge=0.0, le=5.0, description="Window overlap in seconds")


class DiarizationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DIARIZATION_", env_file=".env", extra="ignore")

    enabled: bool = Field(False, description="Enable pyannote speaker diarization")
    hf_auth_token: str = Field("", description="HuggingFace auth token for pyannote")


class ChromaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHROMA_", env_file=".env", extra="ignore")

    persist_dir: Path = Field(Path("./data/chromadb"), description="ChromaDB persistence directory")
    collection: str = Field("debate_transcripts", description="Default collection name")


class OverlaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OVERLAY_", env_file=".env", extra="ignore")

    host: str = Field("0.0.0.0", description="Overlay WebSocket server bind address")
    port: int = Field(8765, description="Overlay WebSocket server port")


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    sqlite_path: Path = Field(Path("./data/debate_watcher.db"))
    fallacy_confidence_threshold: float = Field(0.65, ge=0.0, le=1.0)
    obs_push_confidence_threshold: float = Field(
        0.75, ge=0.0, le=1.0,
        description=(
            "Higher threshold for auto-pushing to OBS without human approval. "
            "Alerts meeting fallacy_confidence_threshold but below this value are "
            "saved to the database for human review before being pushed."
        ),
    )
    vad_silence_threshold: float = Field(0.6, ge=0.0, le=5.0)
    knowledge_base_dir: Path = Field(
        Path("./knowledge_base"),
        description="Directory containing .txt/.md reference documents for fact-checking",
    )
    log_level: str = Field("INFO")
    log_file: Path = Field(Path("./data/debate_watcher.log"))

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v.upper()


class Settings:
    """Aggregated settings object – import and use ``settings`` singleton."""

    def __init__(self) -> None:
        self.obs = OBSSettings()
        self.ollama = OllamaSettings()
        self.whisper = WhisperSettings()
        self.audio = AudioSettings()
        self.diarization = DiarizationSettings()
        self.chroma = ChromaSettings()
        self.overlay = OverlaySettings()
        self.pipeline = PipelineSettings()

    def ensure_data_dirs(self) -> None:
        """Create data directories required at runtime if they do not exist."""
        for path in (
            self.chroma.persist_dir,
            self.pipeline.sqlite_path.parent,
            self.pipeline.log_file.parent,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


settings = Settings()
