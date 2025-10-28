from app.services.youtube_service import YouTubeService

if __name__ == "__main__":
    sample_url = "https://youtu.be/TmbV1mHXXH4?si=SCv0DykE1r4881sI"
    yts = YouTubeService()
    res = yts.download_video(sample_url, task_id="test_video")
    if res.get("status") == "completed":
        print("Download succeeded:", res.get("video_path"))
    else:
        print("Download failed:", res)
