"""
Ollama Service Integration.
Handles client connection, health checking, dynamic model discovery, and error diagnostics.
"""

import json
import urllib.request
import urllib.error
from typing import List, Tuple
import ollama

from helper.logger import logger


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
FALLBACK_MODELS = ["qwen2.5:7b", "qwen2.5:1.5b", "llama3.2:latest", "mistral:latest"]


def check_ollama_status(host: str = DEFAULT_OLLAMA_HOST) -> Tuple[bool, str, List[str]]:
    """
    Check if the Ollama daemon is reachable and retrieve installed models.
    
    Returns:
        (is_online: bool, message: str, models: list[str])
    """
    normalized_host = host.rstrip("/")
    tags_url = f"{normalized_host}/api/tags"

    try:
        req = urllib.request.Request(tags_url, headers={"User-Agent": "CliptyMini/1.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_models = data.get("models", [])
            model_names = [m.get("name") or m.get("model") for m in raw_models if m.get("name") or m.get("model")]
            
            if not model_names:
                return (
                    True,
                    "⚠️ Ollama is running, but no models are downloaded yet. Run 'ollama pull qwen2.5:7b' in terminal.",
                    FALLBACK_MODELS
                )
            
            return (
                True,
                f"✅ Connected to Ollama at {host} ({len(model_names)} model(s) available)",
                model_names
            )

    except urllib.error.URLError as e:
        logger.warning(f"Could not connect to Ollama at {host}: {e}")
        return (
            False,
            f"❌ Ollama is offline or unreachable at {host}. Start it with 'ollama serve'.",
            FALLBACK_MODELS
        )
    except Exception as e:
        logger.warning(f"Error querying Ollama models: {e}")
        return (
            False,
            f"❌ Error communicating with Ollama: {e}",
            FALLBACK_MODELS
        )


def get_available_models(host: str = DEFAULT_OLLAMA_HOST) -> List[str]:
    """Retrieve list of available model names, falling back to popular defaults if offline."""
    _, _, models = check_ollama_status(host)
    return models


def get_ollama_client(host: str = DEFAULT_OLLAMA_HOST) -> ollama.Client:
    """Return configured Ollama Client instance."""
    return ollama.Client(host=host)
