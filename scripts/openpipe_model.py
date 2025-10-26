

from openpipe import OpenAI
from app.core.config import get_settings


settings = get_settings()

OPENPIPE_API_KEY = settings.OPENPIPE_API_KEY
from app.services.transcription_service import TranscriptionService

transcriptionService = TranscriptionService()
transcript= transcriptionService.get_transcript("1")
client = OpenAI(
  openpipe={"api_key": f"{OPENPIPE_API_KEY}"}
)

completion = client.chat.completions.create(
    model="openpipe:olive-papers-take",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant specialized in summarizing YouTube video transcripts."
        },
        {
            "role": "user",
            "content": f"""Given the transcript of a YouTube video, your task is to generate a straight to point and informative summary. \n
             The summary should cover key points, main ideas, and critical information, organized in a coherent and structured way. \n
              Ensure that the summary is not exceed 1000 words.\n
             Make sure that the summary retains the flow and structure of the original transcript while omitting unnecessary details. \n
              The summary should be easy to follow, informative, and structured, highlighting important tips, steps, or insights provided in the transcript.
            \n\nTranscript:  {transcript} """
            }
    ],
    temperature=0,
    openpipe={
        "tags": {
            "prompt_id": "counting",
            "any_key": "any_value"
        }
    },
)

print(completion.choices[0].message)
