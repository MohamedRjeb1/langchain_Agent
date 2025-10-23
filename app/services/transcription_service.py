"""
Audio transcription service using Whisper.
"""
import os
import whisper
from typing import Dict, Any, Optional
from datetime import datetime

from app.core.config import get_settings
from app.models.schemas import ProcessingStatus


class TranscriptionService:
    """Service for transcribing audio files using Whisper."""
    
    def __init__(self):
        self.settings = get_settings()
        self.transcript_dir = self.settings.TRANSCRIPT_DIR
        
        # Ensure transcript directory exists
        os.makedirs(self.transcript_dir, exist_ok=True)
        
        # Load Whisper model
        self.model = whisper.load_model(self.settings.WHISPER_MODEL)
    
    def transcribe_audio(self, audio_file: str, task_id: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_file: Path to audio file
            task_id: Unique task identifier
            language: Language code for transcription
            
        Returns:
            Dict containing transcription information
        """
        try:
            # Use provided language or default from settings
            transcribe_language = language or self.settings.WHISPER_LANGUAGE
            
            # Transcribe the audio
            result = self.model.transcribe(
                audio_file, 
                language=transcribe_language,
                verbose=False
            )
            
            transcript = result["text"].strip()
            detected_language = result.get("language", transcribe_language)
            
            # Save transcript to file
            transcript_file = os.path.join(self.transcript_dir, f"{task_id}_transcript.txt")
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(transcript)
            
            return {
                "task_id": task_id,
                "transcript": transcript,
                "transcript_file": transcript_file,
                "detected_language": detected_language,
                "status": ProcessingStatus.COMPLETED,
                "message": "Transcription completed successfully",
                "word_count": len(transcript.split()),
                "character_count": len(transcript)
            }
            
        except Exception as e:
            return {
                "task_id": task_id,
                "status": ProcessingStatus.FAILED,
                "message": f"Transcription failed: {str(e)}",
                "error": str(e)
            }
    
    def get_transcript(self, task_id: str) -> Optional[str]:
        """
        Retrieve transcript for a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Transcript text or None if not found
        """
        transcript_file = os.path.join(self.transcript_dir, f"{task_id}_transcript.txt")
        
        if os.path.exists(transcript_file):
            with open(transcript_file, "r", encoding="utf-8") as f:
                return f.read()
        
        return None
    
    def delete_transcript(self, task_id: str) -> bool:
        """
        Delete transcript file for a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if deletion successful, False otherwise
        """
        transcript_file = os.path.join(self.transcript_dir, f"{task_id}_transcript.txt")
        
        try:
            if os.path.exists(transcript_file):
                os.remove(transcript_file)
            return True
        except Exception:
            return False
    
    def get_transcript_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get transcript information without loading the full text.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Dict containing transcript metadata or None if not found
        """
        transcript_file = os.path.join(self.transcript_dir, f"{task_id}_transcript.txt")
        
        if not os.path.exists(transcript_file):
            return None
        
        try:
            file_stats = os.stat(transcript_file)
            
            with open(transcript_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            return {
                "task_id": task_id,
                "file_path": transcript_file,
                "file_size": file_stats.st_size,
                "created_at": datetime.fromtimestamp(file_stats.st_ctime),
                "modified_at": datetime.fromtimestamp(file_stats.st_mtime),
                "word_count": len(content.split()),
                "character_count": len(content),
                "line_count": len(content.splitlines())
            }
            
        except Exception:
            return None
