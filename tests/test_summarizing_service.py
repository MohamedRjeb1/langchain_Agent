from app.services.summarizing_service import ModelMistral


if __name__ == "__main__":
    svc = ModelMistral()
    result = svc.summarize_transcript("test1")
    print(result)