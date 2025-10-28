"""
YouTube video download service.
"""
import os
import glob
from typing import Optional, Dict, Any, Callable
import yt_dlp
from yt_dlp import DownloadError

from app.core.config import get_settings
from app.models.schemas import ProcessingStatus


class YouTubeService:
    """Service for downloading YouTube videos."""

    def __init__(self):
        self.settings = get_settings()
        self.output_dir = self.settings.AUDIO_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def download_video(self, url: str, task_id: str, cookiefile: Optional[str] = None, retries: int = 3, progress_hook: Optional[Callable[[dict], None]] = None) -> Dict[str, Any]:
        """
        Download YouTube video as audio file.

        Args:
            url: YouTube video URL
            task_id: Unique task identifier
            cookiefile: Optional path to a cookies.txt file for restricted videos
            retries: Number of retries for transient errors
            progress_hook: Optional callable to receive yt-dlp progress dicts

        Returns:
            Dict containing download information
        """
        ydl_config = {
            "format": self.settings.YOUTUBE_DOWNLOAD_FORMAT if hasattr(self.settings, "YOUTUBE_DOWNLOAD_FORMAT") else "bestaudio[ext=m4a]/bestaudio",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": getattr(self.settings, "YOUTUBE_AUDIO_QUALITY", "192"),
                }
            ],
            # include task_id in filename so we can find it deterministically
            "outtmpl": os.path.join(self.output_dir, f"{task_id}_%(title)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }

        if cookiefile:
            ydl_config["cookiefile"] = cookiefile

        if progress_hook is not None:
            # yt-dlp expects a list of hooks under the "progress_hooks" key
            ydl_config["progress_hooks"] = [progress_hook]

        attempt = 0
        while attempt < retries:
            try:
                with yt_dlp.YoutubeDL(ydl_config) as ydl:
                    info = ydl.extract_info(url, download=False)
                    video_title = info.get("title", "Unknown Title")
                    duration = info.get("duration", 0)

                    # start download (progress will be emitted via progress_hook if provided)
                    ydl.download([url])

                    # Find the downloaded file (mp3 from postprocessor)
                    audio_pattern = os.path.join(self.output_dir, f"{task_id}_*.mp3")
                    audio_files = glob.glob(audio_pattern)

                    if not audio_files:
                        raise DownloadError("No audio file found after download")

                    audio_filename = audio_files[0]

                    return {
                        "task_id": task_id,
                        "video_title": video_title,
                        "duration": duration,
                        "audio_file": audio_filename,
                        "status": ProcessingStatus.COMPLETED,
                        "message": "Video downloaded successfully",
                    }

            except DownloadError as e:
                # unrecoverable download error
                return {
                    "task_id": task_id,
                    "status": ProcessingStatus.FAILED,
                    "message": f"Download failed: {str(e)}",
                    "error": str(e),
                }
            except Exception as e:
                attempt += 1
                if attempt >= retries:
                    return {
                        "task_id": task_id,
                        "status": ProcessingStatus.FAILED,
                        "message": f"Unexpected error after {attempt} attempts: {str(e)}",
                        "error": str(e),
                    }
                # otherwise, retry

    def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        Get video information without downloading.

        Args:
            url: YouTube video URL

        Returns:
            Dict containing video information
        """
        ydl_config = {
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_config) as ydl:
                info = ydl.extract_info(url, download=False)

                return {
                    "title": info.get("title", "Unknown Title"),
                    "duration": info.get("duration", 0),
                    "uploader": info.get("uploader", "Unknown"),
                    "upload_date": info.get("upload_date", "Unknown"),
                    "description": info.get("description", "")[:500] + ("..." if info.get("description") else ""),
                    "thumbnail": info.get("thumbnail", ""),
                    "url": url,
                }

        except Exception as e:
            raise Exception(f"Failed to get video info: {str(e)}")

    def cleanup_audio_file(self, task_id: str) -> bool:
        """
        Clean up audio files for a specific task id.
        Returns True if any file was removed.
        """
        try:
            pattern = os.path.join(self.output_dir, f"{task_id}_*")
            files = glob.glob(pattern)
            for f in files:
                try:
                    os.remove(f)
                except OSError:
                    pass
            return len(files) > 0
        except Exception:
            return False
