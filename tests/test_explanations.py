import copy

import pytest

from predictmaint.config import ExplanationConfig
from predictmaint.service.operator_explanation import operator_explanation, template_explanation


@pytest.fixture
def result():
    return {
        "failure_probability": 0.2,
        "risk_band": "LOW",
        "maintenance_priority": "HIGH",
        "top_drivers": [],
    }


@pytest.mark.parametrize("mode", ["groq", "openai"])
def test_fallback(monkeypatch, result, mode):
    prefix = mode.upper()
    monkeypatch.delenv(f"{prefix}_API_KEY", raising=False)
    monkeypatch.delenv(f"{prefix}_MODEL", raising=False)
    fallback = template_explanation(result)
    assert operator_explanation(result) == fallback
    assert operator_explanation(result, mode) == fallback
    monkeypatch.setenv(f"{prefix}_API_KEY", "test")
    assert operator_explanation(result, mode) == fallback
    monkeypatch.setenv(f"{prefix}_MODEL", "test")

    def fail(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr("predictmaint.service.operator_explanation.httpx.post", fail)
    assert operator_explanation(result, mode) == fallback


@pytest.mark.parametrize(
    "mode,endpoint",
    [
        ("groq", "https://api.groq.com/openai/v1/chat/completions"),
        ("openai", "https://api.openai.com/v1/chat/completions"),
    ],
)
def test_llm_summary(monkeypatch, result, mode, endpoint):
    before = copy.deepcopy(result)
    prefix = mode.upper()
    monkeypatch.setenv(f"{prefix}_API_KEY", "provider-key")
    monkeypatch.setenv(f"{prefix}_MODEL", "provider-model")
    monkeypatch.setenv("OPENAI_API_KEY" if mode == "groq" else "GROQ_API_KEY", "wrong-key")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "Review approved procedures."}}]}

    def post(url, **kwargs):
        assert url == endpoint
        assert kwargs["headers"]["Authorization"] == "Bearer provider-key"
        assert kwargs["json"]["model"] == "provider-model"
        assert kwargs["timeout"] == 10.0
        assert "Never alter probability" in kwargs["json"]["messages"][0]["content"]
        return Response()

    monkeypatch.setattr("predictmaint.service.operator_explanation.httpx.post", post)
    assert operator_explanation(result, mode) == "Review approved procedures."
    assert result == before
    assert ExplanationConfig(mode=mode).mode == mode


@pytest.mark.parametrize(
    "body", [{}, {"choices": []}, {"choices": [{"message": {"content": " "}}]}]
)
def test_groq_bad_response(monkeypatch, result, body):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("GROQ_MODEL", "test")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return body

    monkeypatch.setattr(
        "predictmaint.service.operator_explanation.httpx.post", lambda *a, **k: Response()
    )
    assert operator_explanation(result, "groq") == template_explanation(result)


def test_groq_does_not_use_openai_credentials(monkeypatch, result):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_MODEL", "test")

    def unexpected_call(*args, **kwargs):
        pytest.fail("Groq mode must not call OpenAI")

    monkeypatch.setattr("predictmaint.service.operator_explanation.httpx.post", unexpected_call)
    assert operator_explanation(result, "groq") == template_explanation(result)
