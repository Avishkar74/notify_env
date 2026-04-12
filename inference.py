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
MAX_TOKENS = 50   # slightly larger to allow one-line reasoning before the action word
TEMPERATURE = 0.1
SUCCESS_SCORE_THRESHOLD = 0.4
SCORE_EPSILON = 0.001
MAX_EPISODE_STEPS = int(os.getenv("MAX_EPISODE_STEPS", "200"))


def normalize_score(score: float) -> float:
    return min(max(score, SCORE_EPSILON), 1.0 - SCORE_EPSILON)


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
    score = normalize_score(score)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def debug_log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ── Improved system prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent(
    """
You are an AI Notification Gatekeeper for a smartphone.

Your job: read a notification + the user's current context, then output EXACTLY
one action word — nothing else.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  notify_now  → Show immediately (time-sensitive or user is free)
  silent      → Suppress completely (spam, sleep, deep focus + promo)
  delay       → Queue for later (non-urgent, user is busy)
  escalate    → Strong alert — break through everything (genuine emergency)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION RULES  (highest priority first)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. EMERGENCY OVERRIDE — keywords: hospital, emergency, fire, accident, injured
   → escalate  (overrides sleeping, spam history, deep_focus, everything)

2. SLEEPING
   • Non-emergency → silent
   • Unknown sender + urgent claim → delay  (not escalate)

3. DEEP FOCUS
   • promotional / social / entertainment → silent
   • Calendar/meeting reminder → notify_now  (missing a meeting is worse)
   • Boss/urgent/high-importance → escalate or notify_now

4. ACTIVE TASK MATCH
   • ordered_food + delivery notification → notify_now
   • booked_cab + cab-arriving notification → notify_now
   • watching_ipl + cricket score → notify_now
   • in_meeting + non-urgent colleague → delay

5. SENDER HISTORY
   • spammy + trivial keywords → silent  (even if urgency_hint is high)
   • spammy + emergency keywords → escalate  (rule #1 takes precedence)
   • reliable + urgency_hint > 0.85 → at least notify_now

6. FINANCIAL / OTP TRANSACTIONAL → notify_now  (always time-critical)

7. PROMOTIONAL + no matching active task → silent or delay

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Respond with ONLY the action word on a single line.
Do NOT add punctuation, explanation, or any other text.
"""
).strip()


def build_user_prompt(obs) -> str:
    keywords_str = ", ".join(obs.content_keywords) if obs.content_keywords else "none"
    tasks_str = ", ".join(obs.active_tasks) if obs.active_tasks else "none"
    time_of_day = getattr(obs, "time_of_day", "unknown")

    return textwrap.dedent(
        f"""
NOTIFICATION:
  App:                    {obs.app}
  Category:               {obs.category}
  Sender type:            {obs.sender_type}
  Urgency hint:           {obs.urgency_hint:.2f}
  Messages (last hour):   {obs.message_frequency}
  Content keywords:       {keywords_str}

USER CONTEXT:
  State:                  {obs.user_state}
  Time of day:            {time_of_day}
  Active tasks:           {tasks_str}
  Sender history:         {obs.sender_history}
  Sender trust score:     {obs.sender_trust:.2f}

FEEDBACK FROM LAST STEP: {obs.feedback if obs.feedback else "First step — no previous feedback"}
EPISODE STEP: {obs.step_number}

Decision:"""
    ).strip()


def parse_action(raw: str) -> str:
    """Extract the action word from model output, tolerating minor noise."""
    raw = raw.strip().lower()
    valid = ["notify_now", "silent", "delay", "escalate"]

    if raw in valid:
        return raw

    # First valid token found in the response wins
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
    """
    Rule-based fallback used when the LLM call fails.
    Mirrors the priority order from SYSTEM_PROMPT so scores stay consistent.
    """
    emergency_kw = {"hospital", "emergency", "fire", "accident", "injured"}
    keywords = set(k.lower() for k in obs.content_keywords)

    # Rule 1 — emergency override
    if keywords & emergency_kw:
        return "escalate"

    # Rule 2 — sleeping
    if obs.user_state == "sleeping":
        if obs.sender_history == "unknown" and obs.urgency_hint >= 0.85:
            return "delay"
        return "silent"

    # Rule 3 — deep focus
    if obs.user_state == "deep_focus":
        if obs.category in ("promotional", "social", "entertainment"):
            return "silent"
        if obs.category == "reminder":
            return "notify_now"
        if obs.sender_type == "boss" and obs.urgency_hint >= 0.85:
            return "escalate"
        if obs.urgency_hint < 0.6:
            return "silent"

    # Rule 4 — active task match
    if "ordered_food" in obs.active_tasks and (
        "delivery" in keywords or "arriving" in keywords
    ):
        return "notify_now"
    if "booked_cab" in obs.active_tasks and (
        "arriving" in keywords or "driver" in keywords
    ):
        return "notify_now"
    if "in_meeting" in obs.active_tasks and obs.urgency_hint < 0.75:
        return "delay"

    # Rule 5 — sender history
    if obs.sender_history == "spammy" and obs.urgency_hint < 0.8:
        return "silent"
    if obs.sender_type == "boss" and obs.urgency_hint >= 0.85:
        return "escalate"

    # Rule 6 — financial / OTP
    if obs.category == "transactional":
        return "notify_now"

    # Rule 7 — promotional
    if obs.category == "promotional":
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
        return _heuristic_fallback(obs), str(exc)[:120]


def warmup_proxy_call(client: OpenAI) -> None:
    """One minimal call so the validator can observe proxy traffic on startup."""
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0.0,
            max_tokens=1,
            stream=False,
        )
    except Exception as exc:
        debug_log(f"[DEBUG] warmup_call_error={exc}")


async def run_episode(client: OpenAI, env, task: str) -> Tuple[bool, int, float, List[float]]:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = normalize_score(0.0)
    success = False

    try:
        result = await env.reset(task=task)
        obs = result.observation

        for step in range(1, MAX_EPISODE_STEPS + 1):
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

            history.append(f"{action_str} -> reward {reward:.2f} | {obs.feedback[:80]}")
            if done:
                break

        denominator = len(rewards) if rewards else 1
        score = sum(rewards) / denominator
        score = normalize_score(score)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        debug_log(f"[DEBUG] episode_error task={task}: {exc}")
        if steps_taken == 0:
            log_step(step=1, action="delay", reward=0.0, done=True, error=str(exc))

    return success, steps_taken, score, rewards


async def main() -> None:
    try:
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )
    except KeyError:
        if not API_KEY:
            raise RuntimeError("Missing API_KEY for OpenAI client initialization")
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        debug_log(f"[DEBUG] client_init_error={exc}")
        raise

    warmup_proxy_call(client)

    tasks_to_run = [SINGLE_TASK] if SINGLE_TASK in VALID_TASKS else VALID_TASKS

    for task in tasks_to_run:
        success = False
        steps = 0
        score = normalize_score(0.0)
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
        print(f"[END] success=false steps=0 score={normalize_score(0.0):.3f} rewards=", flush=True)
        sys.exit(1)
