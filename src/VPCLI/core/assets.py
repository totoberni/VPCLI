import os
import pickle
import warnings
from typing import Any, Dict

# This will be the single source of truth for the assets directory path.
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets")

# This cache will be shared by all calls to load_artifact.
_cache: Dict[str, Any] = {}


def load_artifact(filename: str) -> Any:
    """
    Loads a .pkl artifact from the assets directory, with in-memory caching.
    
    Surpresses UserWarning which can be present in older pickle files.
    """
    if filename in _cache:
        return _cache[filename]

    path = os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(path):
        print(f"FATAL ERROR: Artifact '{filename}' not found in '{ASSETS_DIR}'")
        raise FileNotFoundError(f"Missing required asset: {filename}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            with open(path, "rb") as f:
                artifact = pickle.load(f)

        _cache[filename] = artifact
        return artifact
    except Exception as e:
        print(f"ERROR: Failed to load artifact '{filename}'. Reason: {e}")
        raise
