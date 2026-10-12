"""Report whether Ollama is reachable with the configured model pulled."""
import os
import time

import requests

_cache = {}


def ollama_status(ttl: float = 10.0) -> str:
    """Return "llm" if Ollama is up and OLLAMA_MODEL is pulled, else "extractive"."""
    host = os.getenv("OLLAMA_HOST") or "http://localhost:11434"
    if not host.startswith("http"):
        host = "http://" + host
    model = os.getenv("OLLAMA_MODEL") or "llama3"

    cached = _cache.get((host, model))
    if cached and time.monotonic() - cached[0] < ttl:
        return cached[1]
    try:
        resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=1.5)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        ready = any(n == model or n.split(":")[0] == model for n in names)
        mode = "llm" if ready else "extractive"
    except Exception:
        mode = "extractive"
    _cache[(host, model)] = (time.monotonic(), mode)
    return mode
