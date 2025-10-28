"""
Audio transcription service using Whisper.
"""
import os
from typing import Dict, Any, Optional
from datetime import datetime

from app.core.config import get_settings
from app.models.schemas import ProcessingStatus
from app.services.model_loader_service import ModelLoaderService


# module-level placeholder so tests can monkeypatch `whisper` before it's imported
whisper = None


class TranscriptionService:
    """Service for transcribing audio files using Whisper."""

    def __init__(self):
        self.settings = get_settings()
        self.transcript_dir = self.settings.TRANSCRIPT_DIR

        # Ensure transcript directory exists
        os.makedirs(self.transcript_dir, exist_ok=True)

        # model loader service (manages preload and cache)
        self.loader = ModelLoaderService()
        self.model = None

    def transcribe_audio(self, audio_file: str, task_id: str, language: Optional[str] = "en", progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Transcribe audio file to text.

        This method will NOT trigger a model download. It uses ModelLoaderService.get_model
        which returns None if the model is not cached locally. To preload the model, call
        ModelLoaderService.preload_model from an admin action.
        """
        try:
            transcribe_language = language or self.settings.WHISPER_LANGUAGE

            # obtain model from loader; allow download on first transcription so the model
            # is preloaded automatically when transcription starts (cached afterwards)
            if progress_callback:
                progress_callback(f"[transcribe] Loading Whisper model '{self.settings.WHISPER_MODEL}' (may download on first use)")
            # Force loading (and downloading if needed) of the configured model without any fallback
            requested = self.settings.WHISPER_MODEL
            model = self.loader.get_model(model_name=requested, allow_download=True)
            if model is None:
                return {
                    "task_id": task_id,
                    "status": ProcessingStatus.FAILED,
                    "message": (
                        f"Whisper model '{requested}' could not be loaded. No fallback will be attempted.\n"
                        "Tip: Ensure network access and try pre-downloading the model:"
                        " python -c \"import whisper; whisper.load_model('" + requested + "')\""
                    ),
                    "error": "model_not_loaded",
                }
            if progress_callback:
                progress_callback("[transcribe] Whisper model loaded")

            # Transcribe using the loaded model (this will trigger download on first use)
            result = model.transcribe(audio_file, language=transcribe_language, verbose=False)

            transcript = result.get("text", "").strip()
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
                "character_count": len(transcript),
            }

        except Exception as e:
            return {
                "task_id": task_id,
                "status": ProcessingStatus.FAILED,
                "message": f"Transcription failed: {str(e)}",
                "error": str(e),
            }

    def get_transcript(self, task_id: str) -> Optional[str]:
        """
        Retrieve transcript for a specific task.

        Returns transcript text or None if not found.
        """
        transcript_file = os.path.join(self.transcript_dir, f"{task_id}_transcript.txt")

        if os.path.exists(transcript_file):
            with open(transcript_file, "r", encoding="utf-8") as f:
                return f.read()

        return None

    def delete_transcript(self, task_id: str) -> bool:
        """
        Delete transcript file for a specific task.
        """
        transcript_file = os.path.join(self.transcript_dir, f"{task_id}_transcript.txt")

        try:
            if os.path.exists(transcript_file):
                os.remove(transcript_file)
            return True
        except Exception:
            return False

    def get_transcript_info(self, task_id: str) -> Optional[Dict[str, Any]]:
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
                "line_count": len(content.splitlines()),
            }

        except Exception:
            return None

    def _is_model_cached(self, model_name: str) -> bool:
        """Heuristic check for local presence of Whisper model files.

        This checks common cache locations for files that contain the model name
        and are reasonably small. It's a heuristic—if it returns False, the model
        may still be available via other caches, but it's conservative for our needs.
        """
        try:
            user = os.path.expanduser("~")
            candidates = [
                os.path.join(user, ".cache", "whisper"),
                os.path.join(user, ".cache", "huggingface", "hub"),
                os.path.join(user, ".cache"),
            ]
            for base in candidates:
                if not os.path.exists(base):
                    continue
                # walk a few levels deep but bail early when a matching small file found
                for root, dirs, files in os.walk(base):
                    for fn in files:
                        if model_name.lower() in fn.lower():
                            full = os.path.join(root, fn)
                            try:
                                if os.path.getsize(full) > 50 * 1024 * 1024:
                                    return True
                            except Exception:
                                continue
            return False
        except Exception:
            return False

    def preload_model(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Force loading (and downloading if necessary) of a Whisper model.

        Returns a dict with status and time to load. Use this from an admin UI button
        so the user controls when small downloads occur.
        """
        try:
            model_name = model_name or getattr(self.settings, "WHISPER_MODEL", "small")
            whisper_mod = globals().get("whisper", None)
            if whisper_mod is None:
                import whisper as whisper_mod
                globals()["whisper"] = whisper_mod

            import time
            t0 = time.time()
            model = whisper_mod.load_model(model_name)
            dt = time.time() - t0
            # keep it loaded in this service instance
            self.model = model
            self._loaded_model_name = model_name
            return {"status": "completed", "model": model_name, "load_seconds": dt}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
