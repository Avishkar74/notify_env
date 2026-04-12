from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import NotificationAction, NotificationObservation, NotificationState


class NotificationEnv(
    EnvClient[NotificationAction, NotificationObservation, NotificationState]
):
    """Client for the AI Notification Gatekeeper environment."""

    def _step_payload(self, action: NotificationAction) -> Dict:
        return {"decision": action.decision}

    def _parse_result(self, payload: Dict) -> StepResult[NotificationObservation]:
        obs_data = payload.get("observation", {})

        return StepResult(
            observation=NotificationObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                app=obs_data.get("app", ""),
                category=obs_data.get("category", ""),
                sender_type=obs_data.get("sender_type", ""),
                urgency_hint=obs_data.get("urgency_hint", 0.0),
                message_frequency=obs_data.get("message_frequency", 0),
                content_keywords=obs_data.get("content_keywords", []),
                user_state=obs_data.get("user_state", ""),
                active_tasks=obs_data.get("active_tasks", []),
                sender_history=obs_data.get("sender_history", "unknown"),
                sender_trust=obs_data.get("sender_trust", 0.5),
                step_number=obs_data.get("step_number", 0),
                task=obs_data.get("task", ""),
                feedback=obs_data.get("feedback", ""),
                time_of_day=obs_data.get("time_of_day", "afternoon"),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> NotificationState:
        return NotificationState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", ""),
            current_scenario_idx=payload.get("current_scenario_idx", 0),
            total_reward=payload.get("total_reward", 0.0),
            decision_history=payload.get("decision_history", []),
            importance_history=payload.get("importance_history", []),
            episode_score=payload.get("episode_score", 0.0),
        )


# Backward-compatible alias used by scaffold and examples.
NotifyEnv = NotificationEnv
