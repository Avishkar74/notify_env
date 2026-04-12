from types import SimpleNamespace
import importlib.util
from pathlib import Path


def _load_module(module_name: str, file_name: str):
    path = Path(__file__).resolve().parents[1] / file_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


inference = _load_module("inference_module", "inference.py")
ollamainference = _load_module("ollama_module", "ollamainference.py")


def _obs(**kwargs):
    base = {
        "app": "Slack",
        "category": "informative",
        "sender_type": "colleague",
        "urgency_hint": 0.4,
        "message_frequency": 1,
        "content_keywords": ["hello"],
        "user_state": "work",
        "active_tasks": [],
        "sender_history": "responsive",
        "sender_trust": 0.7,
        "step_number": 1,
        "task": "signal_clarity",
        "feedback": "",
        "time_of_day": "afternoon",
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_max_tokens_updated_to_50() -> None:
    assert inference.MAX_TOKENS == 50
    assert ollamainference.MAX_TOKENS == 50


def test_dynamic_step_guard_present() -> None:
    assert inference.MAX_EPISODE_STEPS >= 50
    assert ollamainference.MAX_EPISODE_STEPS >= 50


def test_emergency_override_in_fallback() -> None:
    obs = _obs(
        user_state="sleeping",
        sender_history="spammy",
        content_keywords=["fire", "accident"],
        urgency_hint=0.1,
    )
    assert inference._heuristic_fallback(obs) == "escalate"
    assert ollamainference._heuristic_fallback(obs) == "escalate"


def test_sleeping_unknown_urgent_delays() -> None:
    obs = _obs(user_state="sleeping", sender_history="unknown", urgency_hint=0.9)
    assert inference._heuristic_fallback(obs) == "delay"
    assert ollamainference._heuristic_fallback(obs) == "delay"


def test_deep_focus_promotional_silent() -> None:
    obs = _obs(user_state="deep_focus", category="promotional", sender_history="spammy")
    assert inference._heuristic_fallback(obs) == "silent"
    assert ollamainference._heuristic_fallback(obs) == "silent"


def test_transactional_notify_now() -> None:
    obs = _obs(category="transactional", content_keywords=["otp", "verify"], urgency_hint=0.9)
    assert inference._heuristic_fallback(obs) == "notify_now"
    assert ollamainference._heuristic_fallback(obs) == "notify_now"
