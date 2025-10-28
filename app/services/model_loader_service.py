"""
Service responsible for loading and preloading Whisper models.
This centralizes model downloads and cache checks so the transcription service
doesn't trigger downloads by itself.
"""
import os
from typing import Optional, Dict, Any


class ModelLoaderService:
    """Manage Whisper model loading and preload.

    Methods:
    - is_model_cached(model_name): heuristic check whether weights are present locally
    - preload_model(model_name): force load (and download if needed)
    - get_model(allow_download=False): return loaded model or None; won't download unless allow_download True
    """

    def __init__(self, default_model: Optional[str] = None):
        # default_model is optional; transcription service will pass settings value
        self.default_model = default_model or os.environ.get("WHISPER_MODEL", "large")
        self._model = None
        self._loaded_name = None

    def is_model_cached(self, model_name: Optional[str] = None) -> bool:
        name = model_name or self.default_model
        try:
            user = os.path.expanduser("~")
            candidates = [
                os.path.join(user, ".cache", "whisper"),
                os.path.join(user, ".cache", "huggingface", "hub"),
                os.path.join(user, ".cache"),
            ]
            for base in candidates:
                if not os.path.exists(base):
                    continue
                for root, dirs, files in os.walk(base):
                    for fn in files:
                        if name.lower() in fn.lower():
                            full = os.path.join(root, fn)
                            try:
                                if os.path.getsize(full) > 50 * 1024 * 1024:
                                    return True
                            except Exception:
                                continue
            return False
        except Exception:
            return False

    def preload_model(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        name = model_name or self.default_model
        try:
            import whisper
            import time
            t0 = time.time()
            model = whisper.load_model(name)
            dt = time.time() - t0
            self._model = model
            self._loaded_name = name
            return {"status": "completed", "model": name, "seconds": dt}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def get_model(self, model_name: Optional[str] = None, allow_download: bool = False):
        """Return a loaded model instance or None.

        If a model was preloaded this returns it. If not preloaded and allow_download is False,
        returns None to avoid triggering downloads. If allow_download is True, will call preload_model.
        """
        name = model_name or self.default_model
        if self._model is not None and self._loaded_name == name:
            return self._model

        if not self.is_model_cached(name) and not allow_download:
            return None

        # either cached or downloads allowed
        res = self.preload_model(name)
        if res.get("status") == "completed":
            return self._model
        return None
