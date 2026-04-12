from notify_env.client import NotificationEnv
from notify_env.models import NotificationObservation


def test_observation_time_of_day_default() -> None:
    obs = NotificationObservation(
        done=False,
        reward=None,
        app="Slack",
        category="urgent",
        sender_type="boss",
        urgency_hint=0.9,
        message_frequency=1,
        content_keywords=["urgent"],
        user_state="work",
        active_tasks=["sprint_review"],
        sender_history="reliable",
        sender_trust=0.8,
        step_number=1,
        task="signal_clarity",
        feedback="",
    )
    assert obs.time_of_day == "afternoon"


def test_client_parses_time_of_day_with_default() -> None:
    client = NotificationEnv(base_url="http://localhost:8000")

    payload_with_value = {
        "done": False,
        "reward": 1.0,
        "observation": {
            "app": "Slack",
            "category": "urgent",
            "sender_type": "boss",
            "urgency_hint": 0.95,
            "message_frequency": 1,
            "content_keywords": ["urgent"],
            "user_state": "work",
            "active_tasks": ["sprint_review"],
            "sender_history": "reliable",
            "sender_trust": 0.9,
            "step_number": 2,
            "task": "signal_clarity",
            "feedback": "ok",
            "time_of_day": "morning",
        },
    }
    result = client._parse_result(payload_with_value)
    assert result.observation.time_of_day == "morning"

    payload_without_value = {
        "done": False,
        "reward": 0.5,
        "observation": {
            "app": "Gmail",
            "category": "promotional",
            "sender_type": "marketing",
            "urgency_hint": 0.4,
            "message_frequency": 2,
            "content_keywords": ["offer"],
            "user_state": "relax",
            "active_tasks": [],
            "sender_history": "spammy",
            "sender_trust": 0.2,
            "step_number": 1,
            "task": "signal_clarity",
            "feedback": "",
        },
    }
    result = client._parse_result(payload_without_value)
    assert result.observation.time_of_day == "afternoon"
