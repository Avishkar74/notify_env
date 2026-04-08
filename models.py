from typing import List

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class NotificationAction(Action):
    """Agent decision for a single notification step."""

    decision: str


class NotificationObservation(Observation):
    """Observation payload shown to the agent for each step."""

    app: str
    category: str
    sender_type: str
    urgency_hint: float
    message_frequency: int
    content_keywords: List[str]
    user_state: str
    active_tasks: List[str]
    sender_history: str
    sender_trust: float
    step_number: int
    task: str
    feedback: str


class NotificationState(State):
    """Environment state returned by the state endpoint."""

    task: str = ""
    current_scenario_idx: int = 0
    total_reward: float = 0.0
    decision_history: List[str] = Field(default_factory=list)
    importance_history: List[str] = Field(default_factory=list)
    episode_score: float = 0.0


# Backward-compatible aliases used by the existing scaffold and app wiring.
NotifyAction = NotificationAction
NotifyObservation = NotificationObservation
NotifyState = NotificationState
