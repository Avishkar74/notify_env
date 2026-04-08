import asyncio
import os
import re
import sys
import textwrap
from typing import List, Optional, Tuple

from openai import OpenAI

try:
    from notify_env.client import NotificationEnv
    from notify_env.models import NotificationAction
    from notify_env.server.scenarios import VALID_TASKS
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from notify_env.client import NotificationEnv
    from notify_env.models import NotificationAction
    from notify_env.server.scenarios import VALID_TASKS

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("API_KEY")

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
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


def debug_log(message: str) -> None:
    # Keep validator parsing stable: debug goes to stderr, structured blocks go to stdout.
    print(message, file=sys.stderr, flush=True)


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
    raw = raw.strip().lower()
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


def get_llm_action(client: OpenAI, obs, history: List[str]) -> Tuple[str, Optional[str]]:
    user_prompt = build_user_prompt(obs)

    if history:
        history_block = "\n".join(f"  Step {i + 1}: {h}" for i, h in enumerate(history[-3:]))
        user_prompt = f"RECENT DECISIONS:\n{history_block}\n\n{user_prompt}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw_text = (completion.choices[0].message.content or "").strip()
        return parse_action(raw_text), None
    except Exception as exc:
        return _heuristic_fallback(obs), str(exc)[:100]


async def run_episode(client: OpenAI, env, task: str) -> Tuple[bool, int, float, List[float]]:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = await env.reset(task=task)
        obs = result.observation

        for step in range(1, EPISODE_LENGTH + 1):
            if result.done:
                break

            action_str, llm_error = get_llm_action(client, obs, history)
            result = await env.step(NotificationAction(decision=action_str))

            reward = result.reward if result.reward is not None else 0.0
            done = result.done
            obs = result.observation
            error = getattr(obs, "last_action_error", None)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if llm_error:
                debug_log(f"[DEBUG] model_error={llm_error}")

            history.append(f"{action_str} -> reward {reward:.2f} | {obs.feedback[:60]}")
            if done:
                break

        score = sum(rewards) / EPISODE_LENGTH if EPISODE_LENGTH > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        debug_log(f"[DEBUG] episode_error task={task}: {exc}")
        if steps_taken == 0:
            log_step(
                step=1,
                action="delay",
                reward=0.0,
                done=True,
                error=str(exc),
            )

    return success, steps_taken, score, rewards


async def main() -> None:
    client: OpenAI
    try:
        # Validator requires explicit use of injected proxy vars.
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )
    except KeyError:
        # Local fallback for manual testing outside validator.
        if not API_KEY:
            raise RuntimeError("Missing API_KEY for OpenAI client initialization")
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        debug_log(f"[DEBUG] client_init_error={exc}")
        raise

    tasks_to_run = [SINGLE_TASK] if SINGLE_TASK in VALID_TASKS else VALID_TASKS

    for task in tasks_to_run:
        success = False
        steps = 0
        score = 0.0
        rewards: List[float] = []
        env = None
        log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

        try:
            if LOCAL_IMAGE_NAME:
                env = await NotificationEnv.from_docker_image(LOCAL_IMAGE_NAME)
            else:
                env = NotificationEnv(base_url=ENV_URL)

            success, steps, score, rewards = await run_episode(client, env, task)
        except Exception as exc:
            debug_log(f"[DEBUG] main_task_error task={task}: {exc}")
        finally:
            try:
                if env is not None:
                    await env.close()
            except Exception as err:
                debug_log(f"[DEBUG] env.close_error={err}")
            log_end(success=success, steps=steps, score=score, rewards=rewards)

        debug_log(f"[DEBUG] task={task} score={score:.3f} success={success}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        debug_log(f"[DEBUG] fatal_error={exc}")
        print("[END] success=false steps=0 score=0.000 rewards=", flush=True)
        sys.exit(1)
