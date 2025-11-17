import os
import pickle
import warnings
from typing import Any, Dict

# This cache will be shared across the application to avoid reloading large files.
_asset_cache: Dict[str, Any] = {}

def load_artifact(artifact_name: str) -> Any:
    """
    Loads a data artifact (like a .pkl file) from the assets directory.
    
    This function uses an environment variable `VPCLI_ASSETS_DIR` to locate the
    assets directory, making it compatible with both local development and
    Docker deployments. It falls back to a relative path for local use.
    """
    # --- FIX: Determine the path inside the function at runtime ---
    # This is more robust and ensures the environment variable is always read
    # when the function is called, not when the module is first imported.
    assets_dir = os.getenv(
        "VPCLI_ASSETS_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets")
    )
    
    artifact_path = os.path.join(assets_dir, artifact_name)

    if artifact_name in _asset_cache:
        return _asset_cache[artifact_name]

    if not os.path.exists(artifact_path):
        # Provide a cleaner, more direct error message
        raise FileNotFoundError(
            f"FATAL ERROR: Missing required asset '{artifact_name}' in '{assets_dir}'"
        )
    
    try:
        # --- FIX: Suppress expected UserWarnings from XGBoost ---
        # This prevents warnings about GPU/CPU mismatches from being shown,
        # as they are expected in this application's workflow.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            with open(artifact_path, "rb") as f:
                artifact = pickle.load(f)
                _asset_cache[artifact_name] = artifact
                return artifact
    except Exception as e:
        warnings.warn(f"Could not load artifact {artifact_name}: {e}")
        return None
