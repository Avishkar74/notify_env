from notify_env.models import NotificationAction
from notify_env.server.environment import NotificationEnvironment
from notify_env.server.scenarios import TASK_SCENARIOS


def test_environment_uses_time_of_day_in_observation() -> None:
    env = NotificationEnvironment()
    obs = env.reset(task="signal_clarity")
    assert hasattr(obs, "time_of_day")
    assert obs.time_of_day in {"morning", "afternoon", "evening", "night"}


def test_environment_uses_full_scenario_catalog() -> None:
    env = NotificationEnvironment()
    task = "signal_clarity"
    obs = env.reset(task=task)
    assert obs.task == task
    assert len(env._scenarios) == len(TASK_SCENARIOS[task])
    apps = {s["app"] for s in env._scenarios}
    assert "PhonePe" in apps
    assert "Telegram" in apps


def test_episode_terminates_after_all_task_scenarios() -> None:
    env = NotificationEnvironment()
    task = "signal_clarity"
    env.reset(task=task)
    expected_steps = len(TASK_SCENARIOS[task])

    done = False
    result = None
    for _ in range(expected_steps):
        result = env.step(NotificationAction(decision="delay"))
        done = result.done

    assert result is not None
    assert done is True
    assert result.step_number == expected_steps


def test_notify_now_high_importance_trust_bump() -> None:
    env = NotificationEnvironment()
    env._sender_trust["system"] = 0.8
    scenario = {"sender_type": "system", "importance": "high"}

    env._update_sender_trust("notify_now", scenario, reward=1.0)
    assert env._sender_trust["system"] > 0.8
