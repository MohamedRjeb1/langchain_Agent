import importlib
import os
import tempfile
import types


def test_transcription_service_monkeypatched(monkeypatch, tmp_path):
    ts_mod = importlib.import_module("app.services.transcription_service")

    class DummyModel:
        def transcribe(self, audio_file, language=None, verbose=False):
            return {"text": "this is a fake transcript", "language": language or "en"}

    # Monkeypatch whisper.load_model used in the module
    monkeypatch.setattr(ts_mod, "whisper", types.SimpleNamespace(load_model=lambda m: DummyModel()))

    # Create a fake audio file
    audio_file = tmp_path / "fake_audio.mp3"
    audio_file.write_bytes(b"FAKE")

    svc = ts_mod.TranscriptionService()
    res = svc.transcribe_audio(str(audio_file), task_id="tt1", language="en")

    assert res.get("status") == ts_mod.ProcessingStatus.COMPLETED
    assert "transcript" in res
