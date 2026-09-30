import json
import logging
import os

import httpx


def template_explanation(result: dict) -> str:
    drivers = ", ".join(item["feature"] for item in result["top_drivers"])
    return (
        f"Model failure probability is {result['failure_probability']:.1%}; "
        f"demonstration risk band {result['risk_band']} and review priority "
        f"{result['maintenance_priority']}. Strongest model attributions: "
        f"{drivers or 'unavailable'}. Attribution is not causality. "
        "Review using approved manufacturer maintenance procedures."
    )


def operator_explanation(result: dict, mode: str = "template") -> str:
    fallback = template_explanation(result)
    providers = {
        "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ"),
        "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI"),
    }
    if mode not in providers:
        return fallback
    endpoint, env_prefix = providers[mode]
    key = os.getenv(f"{env_prefix}_API_KEY")
    model = os.getenv(f"{env_prefix}_MODEL")
    if not key or not model:
        return fallback
    try:
        response = httpx.post(
            endpoint,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Summarize only the supplied model outputs in at most 80 "
                            "words. Never alter probability, decision, threshold, risk "
                            "band or priority. Do not invent readings, failure causes, "
                            "repairs, downtime, certainty or economic impact. Attribution "
                            "is not causality. Recommend review under approved procedures, "
                            "never physical repair instructions."
                        ),
                    },
                    {"role": "user", "content": json.dumps(result)},
                ],
            },
            timeout=10.0,
        )
        response.raise_for_status()
        summary = response.json()["choices"][0]["message"]["content"]
        if not isinstance(summary, str) or not summary.strip():
            return fallback
        return summary[:2000]
    except Exception:
        # Do not log HTTP exception content: it can contain authentication headers.
        logging.warning("Optional explanation failed; using deterministic fallback")
        return fallback
