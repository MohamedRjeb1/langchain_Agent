"""
YouTube video download service.
"""
import os
import glob
import uuid
from typing import Optional, Dict, Any
import yt_dlp
from yt_dlp import DownloadError

from app.core.config import get_settings
from app.models.schemas import ProcessingStatus


class YouTubeService:
    """Service for downloading YouTube videos."""
    
    def __init__(self):
        self.settings = get_settings()
        self.output_dir = self.settings.AUDIO_DIR
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def download_video(self, url: str, task_id: str) -> Dict[str, Any]:
        """
        Download YouTube video as audio file.
        
        Args:
            url: YouTube video URL
            task_id: Unique task identifier
            
        Returns:
            Dict containing download information
            
        Raises:
            DownloadError: If download fails
        """
        # Configure yt-dlp options
        ydl_config = {
            "format": self.settings.YOUTUBE_DOWNLOAD_FORMAT,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": self.settings.YOUTUBE_AUDIO_QUALITY,
                }
            ],
            "outtmpl": os.path.join(self.output_dir, f"{task_id}_%(title)s.%(ext)s"),
            "verbose": False,
            "no_warnings": True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_config) as ydl:
                # Extract video info first
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'Unknown Title')
                duration = info.get('duration', 0)
                
                # Download the video
                ydl.download([url])
                
                # Find the downloaded file
                audio_files = glob.glob(os.path.join(self.output_dir, f"{task_id}_*.mp3"))
                
                if not audio_files:
                    raise DownloadError("No audio file found after download")
                
                audio_filename = audio_files[0]
                
                return {
                    "task_id": task_id,
                    "video_title": video_title,
                    "duration": duration,
                    "audio_file": audio_filename,
                    "status": ProcessingStatus.COMPLETED,
                    "message": "Video downloaded successfully"
                }
                
        except DownloadError as e:
            return {
                "task_id": task_id,
                "status": ProcessingStatus.FAILED,
                "message": f"Download failed: {str(e)}",
                "error": str(e)
            }
        except Exception as e:
            return {
                "task_id": task_id,
                "status": ProcessingStatus.FAILED,
                "message": f"Unexpected error: {str(e)}",
                "error": str(e)
            }
    
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
                    "title": info.get('title', 'Unknown Title'),
                    "duration": info.get('duration', 0),
                    "uploader": info.get('uploader', 'Unknown'),
                    "upload_date": info.get('upload_date', 'Unknown'),
                    "view_count": info.get('view_count', 0),
                    "description": info.get('description', '')[:500] + '...' if info.get('description') else '',
                    "thumbnail": info.get('thumbnail', ''),
                    "url": url
                }
                
        except Exception as e:
            raise Exception(f"Failed to get video info: {str(e)}")
    
    def cleanup_audio_file(self, task_id: str) -> bool:
        """
        Clean up audio file for a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            audio_files = glob.glob(os.path.join(self.output_dir, f"{task_id}_*.mp3"))
            for file_path in audio_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            return True
        except Exception:
            return False
