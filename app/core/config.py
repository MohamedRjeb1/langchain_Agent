"""
Configuration settings for the Advanced RAG Application.
"""
import os
from typing import Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    
    
    # Data paths
    DATA_DIR: str = Field(default="data", description="Data directory")
    AUDIO_DIR: str = Field(default="data/audio", description="Audio files directory")
    TRANSCRIPT_DIR: str = Field(default="data/transcripts", description="Transcripts directory")
    VECTORSTORE_DIR: str = Field(default="data/vectorstore", description="Vector store directory")
    
    
    
    # LLM Settings - Modèles locaux
    DEFAULT_LLM_MODEL: str = Field(default="mistral-local", description="Default LLM model")
    DEFAULT_EMBEDDING_MODEL: str = Field(default="nomic-local", description="Default embedding model")
    
    # Whisper settings
    WHISPER_MODEL: str = Field(default="small", description="Whisper model size")
    WHISPER_LANGUAGE: str = Field(default="fr", description="Whisper language")
    
    
    # Vector store settings
    VECTORSTORE_TYPE: str = Field(default="docarray", description="Vector store type")
    EMBEDDING_DIMENSION: int = Field(default=768, description="Embedding dimension")
    
    # YouTube settings
    YOUTUBE_DOWNLOAD_FORMAT: str = Field(default="bestaudio", description="YouTube download format")
    YOUTUBE_AUDIO_QUALITY: str = Field(default="192", description="Audio quality")
    
    @validator("DATA_DIR", "AUDIO_DIR", "TRANSCRIPT_DIR", "VECTORSTORE_DIR")
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        os.makedirs(v, exist_ok=True)
        return v
    

    


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    
    return settings
