import importlib


def test_summarizing_service_monkeypatched(monkeypatch):
    sm_mod = importlib.import_module("app.services.summarizing_service")

    # Monkeypatch TranscriptionService.get_transcript
    class DummyTS:
        def get_transcript(self, transcript_id):
            return "This is a dummy transcript about testing."

    monkeypatch.setattr(sm_mod, "TranscriptionService", lambda: DummyTS())

    # Dummy OllamaLLM
    class DummyLLM:
        def __init__(self, model=None):
            self.model = model

        def invoke(self, prompt):
            return "DUMMY SUMMARY"

    monkeypatch.setattr(sm_mod, "OllamaLLM", DummyLLM)

    summarizer = sm_mod.ModelMistral(model_name="mistral-test")
    out = summarizer.summarize_transcript("1")
    assert isinstance(out, str)
    assert out == "DUMMY SUMMARY"
