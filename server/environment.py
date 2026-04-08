import uuid
from typing import Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment

from notify_env.models import NotificationAction, NotificationObservation, NotificationState
from notify_env.server.scenarios import TASK_SCENARIOS, VALID_TASKS

REWARD_PERFECT = 1.0
REWARD_ACCEPTABLE = 0.5
REWARD_WRONG = 0.0

TRUST_INCREASE = 0.15
TRUST_DECREASE = 0.20


class NotificationEnvironment(Environment):
    """AI Notification Gatekeeper environment with 3 tasks and 5-step episodes."""

    SUPPORTS_CONCURRENT_SESSIONS = True
    EPISODE_LENGTH = 5

    def __init__(self):
        self._state: NotificationState = NotificationState()
        self._task: str = "signal_clarity"
        self._scenarios: List[dict] = []
        self._current_step: int = 0
        self._total_reward: float = 0.0
        self._decision_history: List[str] = []
        self._importance_history: List[str] = []
        self._sender_trust: Dict[str, float] = {}

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> NotificationObservation:
        self._task = task if task in VALID_TASKS else "signal_clarity"
        self._scenarios = TASK_SCENARIOS[self._task]

        self._current_step = 0
        self._total_reward = 0.0
        self._decision_history = []
        self._importance_history = []
        self._sender_trust = {}

        self._state = NotificationState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task=self._task,
            current_scenario_idx=0,
            total_reward=0.0,
            decision_history=[],
            importance_history=[],
            episode_score=0.0,
        )

        scenario = self._scenarios[0]
        sender = scenario["sender_type"]
        initial_trust = self._get_initial_trust(scenario["sender_history"])
        self._sender_trust[sender] = initial_trust

        return NotificationObservation(
            done=False,
            reward=None,
            app=scenario["app"],
            category=scenario["category"],
            sender_type=scenario["sender_type"],
            urgency_hint=scenario["urgency_hint"],
            message_frequency=scenario["message_frequency"],
            content_keywords=scenario["content_keywords"],
            user_state=scenario["user_state"],
            active_tasks=scenario["active_tasks"],
            sender_history=scenario["sender_history"],
            sender_trust=initial_trust,
            step_number=1,
            task=self._task,
            feedback="",
        )

    def step(
        self,
        action: NotificationAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> NotificationObservation:
        # Guard against direct /step calls before /reset in stateless HTTP usage.
        if not self._scenarios or self._current_step >= len(self._scenarios):
            self.reset(task=self._task)

        valid_actions = {"notify_now", "silent", "delay", "escalate"}
        decision = action.decision if action.decision in valid_actions else "delay"

        self._state.step_count += 1
        current_scenario = self._scenarios[self._current_step]

        reward, feedback = self._compute_reward(decision, current_scenario)
        self._update_sender_trust(decision, current_scenario, reward)

        self._total_reward += reward
        self._decision_history.append(decision)
        self._importance_history.append(current_scenario["importance"])
        self._current_step += 1

        done = self._current_step >= self.EPISODE_LENGTH

        episode_score = self._total_reward / self.EPISODE_LENGTH
        self._state = NotificationState(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            task=self._task,
            current_scenario_idx=self._current_step,
            total_reward=round(self._total_reward, 4),
            decision_history=list(self._decision_history),
            importance_history=list(self._importance_history),
            episode_score=round(episode_score, 4),
        )

        if done:
            return NotificationObservation(
                done=True,
                reward=reward,
                app="",
                category="",
                sender_type="",
                urgency_hint=0.0,
                message_frequency=0,
                content_keywords=[],
                user_state="",
                active_tasks=[],
                sender_history="",
                sender_trust=0.0,
                step_number=self._current_step,
                task=self._task,
                feedback=(
                    f"{feedback} | Episode complete. "
                    f"Score: {episode_score:.2f} "
                    f"({self._total_reward:.1f}/{self.EPISODE_LENGTH:.1f})"
                ),
            )

        next_scenario = self._scenarios[self._current_step]
        next_sender = next_scenario["sender_type"]
        if next_sender not in self._sender_trust:
            self._sender_trust[next_sender] = self._get_initial_trust(
                next_scenario["sender_history"]
            )
        next_trust = self._sender_trust[next_sender]

        return NotificationObservation(
            done=False,
            reward=reward,
            app=next_scenario["app"],
            category=next_scenario["category"],
            sender_type=next_scenario["sender_type"],
            urgency_hint=next_scenario["urgency_hint"],
            message_frequency=next_scenario["message_frequency"],
            content_keywords=next_scenario["content_keywords"],
            user_state=next_scenario["user_state"],
            active_tasks=next_scenario["active_tasks"],
            sender_history=next_scenario["sender_history"],
            sender_trust=next_trust,
            step_number=self._current_step + 1,
            task=self._task,
            feedback=feedback,
        )

    @property
    def state(self) -> NotificationState:
        return self._state

    def _compute_reward(self, decision: str, scenario: dict) -> Tuple[float, str]:
        expected = scenario["expected_action"]
        acceptable = scenario.get("acceptable_action")

        if decision == expected:
            return REWARD_PERFECT, scenario["feedback_correct"]

        if acceptable and decision == acceptable:
            return (
                REWARD_ACCEPTABLE,
                f"Acceptable but not optimal. {scenario['feedback_correct']}",
            )

        return REWARD_WRONG, scenario["feedback_wrong"]

    def _update_sender_trust(
        self,
        decision: str,
        scenario: dict,
        reward: float,
    ) -> None:
        sender = scenario["sender_type"]
        importance = scenario["importance"]
        current_trust = self._sender_trust.get(sender, 0.7)

        if decision == "escalate" and importance == "urgent" and reward == REWARD_PERFECT:
            self._sender_trust[sender] = min(1.0, current_trust + TRUST_INCREASE)
        elif decision == "silent" and importance in ("urgent", "high") and reward == REWARD_WRONG:
            self._sender_trust[sender] = max(0.0, current_trust - TRUST_DECREASE)
        elif decision == "escalate" and importance == "low" and reward == REWARD_WRONG:
            self._sender_trust[sender] = max(0.0, current_trust - TRUST_DECREASE * 0.5)

    def _get_initial_trust(self, sender_history: str) -> float:
        trust_map = {
            "reliable": 0.85,
            "responsive": 0.70,
            "unknown": 0.50,
            "spammy": 0.25,
        }
        return trust_map.get(sender_history, 0.50)
