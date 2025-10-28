from app.services.transcription_service import TranscriptionService


if __name__ == "__main__":
    svc = TranscriptionService()
    result = svc.transcribe_audio(r"C:\Users\moham\OneDrive\Desktop\la\data\audio\test_video_YouTube MCP Server ： AI for YouTube.mp3", task_id="test1", language="en")
    print(result)