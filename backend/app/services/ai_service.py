"""Server-only AI provider boundary with a deterministic, grounded fallback."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any
from app.config import settings
from app.services.platform_service import local_copilot


def answer(question: str, workflow: dict[str, Any]) -> dict[str, str]:
    """Use an explicitly configured OpenAI-compatible provider, or fall back safely.

    The complete numeric context is supplied to the server-side model and the
    response is labelled as provider output. Provider failures never block the
    demo flow and are intentionally not surfaced with implementation details.
    """
    if settings.AI_PROVIDER.lower() != "openai" or not settings.AI_API_KEY:
        return {**local_copilot(question, workflow), "provider": "Local Fallback"}
    context = json.dumps({"scenario": workflow["scenario"], "metrics": workflow["metrics"], "affected_settlements": workflow["affected_settlements"], "assignments": workflow["assignments"]}, separators=(",", ":"))
    body = {"model": settings.AI_MODEL, "temperature": 0, "max_tokens": 350, "messages": [{"role": "system", "content": "You are PRISM, a decision-support assistant. Use only supplied data. State that values are simulated demo outputs. If information is absent, say so. Never give operational emergency instructions."}, {"role": "user", "content": f"Question: {question}\nCurrent workflow JSON: {context}"}]}
    request = urllib.request.Request(settings.AI_BASE_URL.rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"), headers={"Authorization": f"Bearer {settings.AI_API_KEY}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = str(payload["choices"][0]["message"]["content"]).strip()
        if not text:
            raise ValueError("empty provider response")
        return {"mode": "configured_ai_provider", "provider": "Configured AI Provider", "answer": text[:4000], "grounding": "Provider response was supplied the current structured workflow context only."}
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, json.JSONDecodeError):
        return {**local_copilot(question, workflow), "provider": "Local Fallback", "provider_notice": "Configured provider was unavailable; deterministic local fallback used."}
