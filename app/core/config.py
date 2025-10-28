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
    OPENPIPE_API_KEY: str = Field(default="open_pipe_api_key", description="OpenPipe API key")
    
    # Whisper settings
    WHISPER_MODEL: str = Field(default="small", description="Whisper model size")
    WHISPER_LANGUAGE: str = Field(default="en", description="Whisper language")
    
    
    # Vector store settings
    VECTORSTORE_TYPE: str = Field(default="docarray", description="Vector store type")
    # Use the project's existing data folder (matches workspace 'data/VectorStore')
    VECTORSTORE_DIR: str = Field(default="data/VectorStore", description="Vector store directory")
    EMBEDDING_DIMENSION: int = Field(default=768, description="Embedding dimension")
    # Chunking defaults (used by indexing / chunking services)
    CHUNK_SIZE: int = Field(default=512, description="Default chunk size in tokens/words")
    CHUNK_OVERLAP: int = Field(default=64, description="Default chunk overlap in tokens/words")
    
    # YouTube settings
    YOUTUBE_DOWNLOAD_FORMAT: str = Field(default="bestaudio", description="YouTube download format")
    YOUTUBE_AUDIO_QUALITY: str = Field(default="192", description="Audio quality")
    
   
    

    


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    
    return settings
