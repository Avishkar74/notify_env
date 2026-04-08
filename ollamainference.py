import asyncio
import json
import os
import re
import sys
import textwrap
import urllib.request
from typing import List, Optional, Tuple

try:
    from notify_env.client import NotificationEnv
    from notify_env.models import NotificationAction
    from notify_env.server.scenarios import VALID_TASKS
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from notify_env.client import NotificationEnv
    from notify_env.models import NotificationAction
    from notify_env.server.scenarios import VALID_TASKS

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

IMAGE_NAME = os.getenv("IMAGE_NAME")
ENV_URL = os.getenv("NOTIF_ENV_URL", "http://localhost:8000")
BENCHMARK = "notify_env"

SINGLE_TASK = os.getenv("NOTIF_TASK")

EPISODE_LENGTH = 5
MAX_TOKENS = 20
TEMPERATURE = 0.1
SUCCESS_SCORE_THRESHOLD = 0.4


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = textwrap.dedent(
    """
You are an AI Notification Gatekeeper.

You receive information about a smartphone notification and the user's current
context. Your job is to decide how to handle it.

AVAILABLE ACTIONS (output EXACTLY one, nothing else):
  notify_now
  silent
  delay
  escalate

CRITICAL OVERRIDES:
  - sleeping + non-urgent = always silent
  - deep_focus + promotional = always silent
  - active_tasks contains "ordered_food" + delivery notification = notify_now
  - sender_history = spammy + trivial content_keywords = silent
  - genuine emergency keywords (hospital, emergency, fire) override spam history

OUTPUT FORMAT:
Respond with ONLY the action word.
"""
).strip()


def build_user_prompt(obs) -> str:
    keywords_str = ", ".join(obs.content_keywords) if obs.content_keywords else "none"
    tasks_str = ", ".join(obs.active_tasks) if obs.active_tasks else "none"

    return textwrap.dedent(
        f"""
NOTIFICATION:
  App: {obs.app}
  Category: {obs.category}
  Sender type: {obs.sender_type}
  Urgency hint: {obs.urgency_hint:.2f}
  Messages from sender in last hour: {obs.message_frequency}
  Content keywords: {keywords_str}

USER CONTEXT:
  Current state: {obs.user_state}
  Active tasks: {tasks_str}
  Sender history: {obs.sender_history}
  Sender trust score: {obs.sender_trust:.2f}

PREVIOUS STEP FEEDBACK: {obs.feedback if obs.feedback else "First step - no previous feedback"}
EPISODE STEP: {obs.step_number} / 5

What is your decision?
"""
    ).strip()


def parse_action(raw: str) -> str:
    raw = (raw or "").strip().lower()
    valid = ["notify_now", "silent", "delay", "escalate"]

    if raw in valid:
        return raw

    for action in valid:
        if action in raw:
            return action

    clean = re.sub(r"[^a-z_\s]", "", raw).strip()
    if clean in valid:
        return clean
    clean_no_space = clean.replace(" ", "_")
    if clean_no_space in valid:
        return clean_no_space

    return "delay"


def _heuristic_fallback(obs) -> str:
    if obs.user_state == "sleeping":
        if "emergency" in obs.content_keywords or "hospital" in obs.content_keywords:
            return "escalate"
        return "silent"

    if obs.category == "promotional" or obs.sender_history == "spammy":
        if obs.urgency_hint < 0.8:
            return "silent"

    if obs.sender_type == "boss" and obs.urgency_hint > 0.8:
        return "escalate"

    if obs.category == "transactional":
        return "notify_now"

    if "ordered_food" in obs.active_tasks and "delivery" in obs.content_keywords:
        return "notify_now"

    if obs.user_state == "deep_focus" and obs.urgency_hint < 0.6:
        return "silent"

    return "delay"


def get_ollama_action(obs, history: List[str]) -> Tuple[str, Optional[str]]:
    user_prompt = build_user_prompt(obs)

    if history:
        history_block = "\n".join(f"  Step {i + 1}: {h}" for i, h in enumerate(history[-3:]))
        user_prompt = f"RECENT DECISIONS:\n{history_block}\n\n{user_prompt}"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }

    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        raw_text = ((result.get("message") or {}).get("content") or "").strip()
        return parse_action(raw_text), None
    except Exception as exc:
        return _heuristic_fallback(obs), str(exc)[:100]


async def run_episode(env, task: str) -> Tuple[bool, int, float, List[float]]:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=OLLAMA_MODEL)

    try:
        result = await env.reset(task=task)
        obs = result.observation

        for step in range(1, EPISODE_LENGTH + 1):
            if result.done:
                break

            action_str, error = get_ollama_action(obs, history)
            result = await env.step(NotificationAction(decision=action_str))

            reward = result.reward if result.reward is not None else 0.0
            done = result.done
            obs = result.observation

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"{action_str} -> reward {reward:.2f} | {obs.feedback[:60]}")
            if done:
                break

        score = sum(rewards) / EPISODE_LENGTH if EPISODE_LENGTH > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        print(f"[DEBUG] Episode error for task={task}: {exc}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return success, steps_taken, score, rewards


async def main() -> None:
    tasks_to_run = [SINGLE_TASK] if SINGLE_TASK in VALID_TASKS else VALID_TASKS

    for task in tasks_to_run:
        if IMAGE_NAME:
            env = await NotificationEnv.from_docker_image(IMAGE_NAME)
        else:
            env = NotificationEnv(base_url=ENV_URL)

        try:
            success, steps, score, rewards = await run_episode(env, task)
        finally:
            try:
                await env.close()
            except Exception as err:
                print(f"[DEBUG] env.close() error: {err}", flush=True)

        print(f"[DEBUG] Task={task} | Score={score:.3f} | Success={success}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())