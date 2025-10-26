import importlib
import os


def test_youtube_get_info_and_cleanup(monkeypatch, tmp_path):
    yt_mod = importlib.import_module("app.services.youtube_service")

    class DummyYDL:
        def __init__(self, cfg=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            return {"title": "Test Video", "duration": 12}

    monkeypatch.setattr(yt_mod, "yt_dlp", type("X", (), {"YoutubeDL": lambda cfg=None: DummyYDL(cfg)}))

    svc = yt_mod.YouTubeService()
    info = svc.get_video_info("https://youtube.test/video")
    assert isinstance(info, dict)
    assert info.get("title") == "Test Video"

    # Create a fake audio file and test cleanup
    audio_dir = svc.output_dir
    os.makedirs(audio_dir, exist_ok=True)
    fake_file = os.path.join(audio_dir, "cleanup_task_test.mp3")
    with open(fake_file, "wb") as f:
        f.write(b"FAKE")

    # Ensure file exists
    assert os.path.exists(fake_file)
    # Call cleanup (uses pattern matching by task id); create matching name
    # Our cleanup helper expects files matching task id prefix; create such file
    prefixed = os.path.join(audio_dir, "task123_Title.mp3")
    with open(prefixed, "wb") as f:
        f.write(b"FAKE")

    assert svc.cleanup_audio_file("task123") is True
