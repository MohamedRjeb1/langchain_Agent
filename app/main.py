

from app.services.youtube_service import YouTubeService
from app.services.transcription_service import TranscriptionService
from app.services.summarising_service import SummarisingService


youtube_service = YouTubeService()
transcriptionService = TranscriptionService()

summarisingService = SummarisingService()




if __name__ == "__main__":
    chunk = transcriptionService.get_transcript("1")
    summary = summarisingService.summarize_chunks([chunk])
    print(summary)
    


