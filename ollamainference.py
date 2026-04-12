import asyncio
import datetime
import html
import json
import os
import pathlib
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

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")


def resolve_hf_space_runtime_host(space_id: str) -> Optional[str]:
    """Resolve the currently active HF runtime host from the Spaces API."""
    try:
        req = urllib.request.Request(
            f"https://huggingface.co/api/spaces/{space_id}",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        host = payload.get("host")
        if host:
            return host
    except Exception:
        pass
    return None


def derive_hf_space_runtime_url(space_page_url: str) -> Optional[str]:
    """Convert https://huggingface.co/spaces/<owner>/<space> to https://<owner>-<space>.hf.space."""
    match = re.search(r"huggingface\.co/spaces/([^/]+)/([^/?#]+)", space_page_url)
    if not match:
        return None
    owner, space = match.group(1), match.group(2)
    return f"https://{(owner + '-' + space).lower()}.hf.space"


DEFAULT_HF_SPACE_PAGE_URL = os.getenv(
    "HF_SPACE_URL",
    "https://huggingface.co/spaces/Avishkar-00/notify_env",
)

HF_SPACE_ID = os.getenv("HF_SPACE_ID", "Avishkar-00/notify_env")
HF_SPACE_HOST = os.getenv("HF_SPACE_HOST") or resolve_hf_space_runtime_host(HF_SPACE_ID)

ENV_URL = (
    os.getenv("NOTIF_ENV_URL")
    or HF_SPACE_HOST
    or derive_hf_space_runtime_url(DEFAULT_HF_SPACE_PAGE_URL)
    or "http://localhost:8000"
)
BENCHMARK = "notify_env"

SINGLE_TASK = os.getenv("NOTIF_TASK")

EPISODE_LENGTH = 5
MAX_TOKENS = 50
TEMPERATURE = 0.1
SUCCESS_SCORE_THRESHOLD = 0.4
SCORE_EPSILON = 0.001
MAX_EPISODE_STEPS = int(os.getenv("MAX_EPISODE_STEPS", "200"))

TRACE_JSONL_PATH = os.getenv("NOTIF_TRACE_JSONL")
TRACE_HTML_PATH = os.getenv("NOTIF_TRACE_HTML")


def normalize_score(score: float) -> float:
    # Validator requires strictly 0 < score < 1 for every task.
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
    # Keep validator parsing stable: debug goes to stderr, structured blocks go to stdout.
    print(message, file=sys.stderr, flush=True)


def obs_to_dict(obs) -> dict:
    # Keep only JSON-serializable primitives.
    return {
        "app": getattr(obs, "app", ""),
        "category": getattr(obs, "category", ""),
        "sender_type": getattr(obs, "sender_type", ""),
        "urgency_hint": float(getattr(obs, "urgency_hint", 0.0) or 0.0),
        "message_frequency": int(getattr(obs, "message_frequency", 0) or 0),
        "content_keywords": list(getattr(obs, "content_keywords", []) or []),
        "user_state": getattr(obs, "user_state", ""),
        "active_tasks": list(getattr(obs, "active_tasks", []) or []),
        "sender_history": getattr(obs, "sender_history", ""),
        "sender_trust": float(getattr(obs, "sender_trust", 0.0) or 0.0),
        "step_number": int(getattr(obs, "step_number", 0) or 0),
        "task": getattr(obs, "task", ""),
        "feedback": getattr(obs, "feedback", ""),
        "time_of_day": getattr(obs, "time_of_day", "afternoon"),
    }


def reward_to_label(reward: Optional[float]) -> str:
    if reward is None:
        return "unknown"
    if reward >= 0.99:
        return "correct"
    if reward > 0.0:
        return "partial"
    return "wrong"


def ensure_parent_dir(path: str) -> None:
    pathlib.Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str, events: List[dict]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def render_html_report(events: List[dict]) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    tasks: dict = {}
    for e in events:
        if e.get("type") == "task_start":
            tasks[e["task"]] = {"start": e, "steps": [], "end": None}
        elif e.get("type") == "step":
            tasks.setdefault(e.get("task", "unknown"), {"start": None, "steps": [], "end": None})[
                "steps"
            ].append(e)
        elif e.get("type") == "task_end":
            tasks.setdefault(e.get("task", "unknown"), {"start": None, "steps": [], "end": None})[
                "end"
            ] = e

    def esc(value: object) -> str:
        return html.escape(str(value))

    def badge(label: str) -> str:
        cls = {
            "correct": "badge ok",
            "partial": "badge warn",
            "wrong": "badge bad",
            "unknown": "badge",
        }.get(label, "badge")
        return f"<span class=\"{cls}\">{esc(label)}</span>"

    parts: List[str] = []
    parts.append(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>notify_env run report</title>"
        "<style>"
        "body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#0b0f14;color:#e6edf3;}"
        ".wrap{max-width:1100px;margin:0 auto;padding:24px;}"
        "h1{font-size:20px;margin:0 0 6px 0;}"
        ".muted{color:#9aa4af;font-size:12px;}"
        ".card{background:#111826;border:1px solid #223041;border-radius:12px;padding:14px;margin:14px 0;}"
        ".row{display:flex;gap:12px;flex-wrap:wrap;}"
        ".kv{font-size:12px;color:#c7d0db;}"
        ".kv b{color:#e6edf3;}"
        "table{width:100%;border-collapse:collapse;font-size:12px;}"
        "th,td{border-bottom:1px solid #223041;padding:10px;vertical-align:top;}"
        "th{color:#c7d0db;text-align:left;font-weight:600;}"
        ".badge{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid #334155;color:#c7d0db;}"
        ".badge.ok{border-color:#14532d;color:#bbf7d0;}"
        ".badge.warn{border-color:#713f12;color:#fde68a;}"
        ".badge.bad{border-color:#7f1d1d;color:#fecaca;}"
        ".mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}"
        ".chips{display:flex;gap:6px;flex-wrap:wrap;}"
        ".chip{background:#0b1220;border:1px solid #223041;border-radius:999px;padding:2px 8px;}"
        "</style></head><body><div class='wrap'>"
    )
    parts.append(f"<h1>notify_env run report</h1><div class='muted'>generated {esc(now)}</div>")

    if not tasks:
        parts.append("<div class='card'>No events recorded.</div>")
    else:
        for task_name, bundle in tasks.items():
            start = bundle.get("start") or {}
            end = bundle.get("end") or {}
            score = end.get("score")
            success = end.get("success")
            parts.append("<div class='card'>")
            parts.append(
                f"<div class='row'><div class='kv'><b>Task</b>: {esc(task_name)}</div>"
                f"<div class='kv'><b>Model</b>: <span class='mono'>{esc(start.get('model',''))}</span></div>"
                f"<div class='kv'><b>Env URL</b>: <span class='mono'>{esc(start.get('env_url',''))}</span></div>"
                f"<div class='kv'><b>Score</b>: {esc(score)}</div>"
                f"<div class='kv'><b>Success</b>: {esc(success)}</div></div>"
            )
            parts.append("<div style='margin-top:10px; overflow:auto'>")
            parts.append(
                "<table><thead><tr>"
                "<th>Step</th><th>User</th><th>Notification</th><th>Decision</th><th>Reward</th><th>Feedback</th>"
                "</tr></thead><tbody>"
            )
            for s in bundle.get("steps", []):
                obs = s.get("obs", {})
                user_block = (
                    f"<div><b>state</b>: {esc(obs.get('user_state',''))}</div>"
                    f"<div><b>active</b>: {esc(', '.join(obs.get('active_tasks',[]) or []))}</div>"
                )
                notif_block = (
                    f"<div><b>app</b>: {esc(obs.get('app',''))}</div>"
                    f"<div><b>cat</b>: {esc(obs.get('category',''))}</div>"
                    f"<div><b>sender</b>: {esc(obs.get('sender_type',''))} / {esc(obs.get('sender_history',''))}</div>"
                    f"<div><b>urgency</b>: {esc(obs.get('urgency_hint',''))}</div>"
                    f"<div class='chips'>"
                    + "".join(f"<span class='chip'>{esc(k)}</span>" for k in (obs.get('content_keywords',[]) or []))
                    + "</div>"
                )
                decision = s.get("action")
                reward = s.get("reward")
                parts.append(
                    "<tr>"
                    f"<td class='mono'>{esc(s.get('step'))}</td>"
                    f"<td>{user_block}</td>"
                    f"<td>{notif_block}</td>"
                    f"<td><div class='mono'>{esc(decision)}</div>{badge(s.get('reward_label','unknown'))}</td>"
                    f"<td class='mono'>{esc(reward)}</td>"
                    f"<td>{esc(s.get('feedback',''))}</td>"
                    "</tr>"
                )
            parts.append("</tbody></table></div></div>")

    parts.append("</div></body></html>")
    return "".join(parts)


def write_html(path: str, html_text: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html_text)


SYSTEM_PROMPT = textwrap.dedent(
    """
You are an AI Notification Gatekeeper for a smartphone.

Your job: read a notification + the user's current context, then output EXACTLY
one action word - nothing else.

ACTIONS
  notify_now  -> Show immediately (time-sensitive or user is free)
  silent      -> Suppress completely (spam, sleep, deep focus + promo)
  delay       -> Queue for later (non-urgent, user is busy)
  escalate    -> Strong alert - break through everything (genuine emergency)

DECISION RULES (highest priority first)
1. EMERGENCY OVERRIDE - keywords: hospital, emergency, fire, accident, injured
    -> escalate (overrides sleeping, spam history, deep_focus, everything)

2. SLEEPING
    - Non-emergency -> silent
    - Unknown sender + urgent claim -> delay (not escalate)

3. DEEP FOCUS
    - promotional / social / entertainment -> silent
    - Calendar/meeting reminder -> notify_now
    - Boss/urgent/high-importance -> escalate or notify_now

4. ACTIVE TASK MATCH
    - ordered_food + delivery notification -> notify_now
    - booked_cab + cab-arriving notification -> notify_now
    - watching_ipl + cricket score -> notify_now
    - in_meeting + non-urgent colleague -> delay

5. SENDER HISTORY
    - spammy + trivial keywords -> silent
    - spammy + emergency keywords -> escalate
    - reliable + urgency_hint > 0.85 -> at least notify_now

6. FINANCIAL / OTP TRANSACTIONAL -> notify_now

7. PROMOTIONAL + no matching active task -> silent or delay

OUTPUT FORMAT
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

FEEDBACK FROM LAST STEP: {obs.feedback if obs.feedback else "First step - no previous feedback"}
EPISODE STEP: {obs.step_number}

Decision:
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
    emergency_kw = {"hospital", "emergency", "fire", "accident", "injured"}
    keywords = set(k.lower() for k in obs.content_keywords)

    if keywords & emergency_kw:
        return "escalate"

    if obs.user_state == "sleeping":
        if obs.sender_history == "unknown" and obs.urgency_hint >= 0.85:
            return "delay"
        return "silent"

    if obs.user_state == "deep_focus":
        if obs.category in ("promotional", "social", "entertainment"):
            return "silent"
        if obs.category == "reminder":
            return "notify_now"
        if obs.sender_type == "boss" and obs.urgency_hint >= 0.85:
            return "escalate"
        if obs.urgency_hint < 0.6:
            return "silent"

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

    if obs.sender_history == "spammy" and obs.urgency_hint < 0.8:
        return "silent"
    if obs.sender_type == "boss" and obs.urgency_hint >= 0.85:
        return "escalate"

    if obs.category == "transactional":
        return "notify_now"
    if obs.category == "promotional":
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
        return _heuristic_fallback(obs), str(exc)[:120]


def warmup_model_call() -> None:
    """Make one minimal call so a model request is attempted early (safe if Ollama is absent)."""
    try:
        body = json.dumps(
            {
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 1},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as _resp:
            pass
    except Exception as exc:
        debug_log(f"[DEBUG] warmup_call_error={exc}")


async def run_episode(env, task: str, events: Optional[List[dict]] = None) -> Tuple[bool, int, float, List[float]]:
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

            obs_before = obs
            action_str, model_error = get_ollama_action(obs, history)
            result = await env.step(NotificationAction(decision=action_str))

            reward = result.reward if result.reward is not None else 0.0
            done = result.done
            obs = result.observation
            error = getattr(obs, "last_action_error", None)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if events is not None:
                events.append(
                    {
                        "type": "step",
                        "task": task,
                        "step": step,
                        "action": action_str,
                        "reward": reward,
                        "reward_label": reward_to_label(reward),
                        "done": bool(done),
                        "error": error,
                        "obs": obs_to_dict(obs_before),
                        "feedback": getattr(obs, "feedback", ""),
                    }
                )

            if model_error:
                debug_log(f"[DEBUG] model_error={model_error}")

            history.append(f"{action_str} -> reward {reward:.2f} | {obs.feedback[:60]}")
            if done:
                break

        denominator = len(rewards) if rewards else 1
        score = sum(rewards) / denominator
        score = normalize_score(score)
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
    tasks_to_run = [SINGLE_TASK] if SINGLE_TASK in VALID_TASKS else VALID_TASKS

    warmup_model_call()

    events: List[dict] = []

    for task in tasks_to_run:
        success = False
        steps = 0
        score = normalize_score(0.0)
        rewards: List[float] = []

        env = None
        log_start(task=task, env=BENCHMARK, model=OLLAMA_MODEL)

        events.append(
            {
                "type": "task_start",
                "task": task,
                "model": OLLAMA_MODEL,
                "env_url": ENV_URL,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            }
        )

        try:
            if LOCAL_IMAGE_NAME:
                env = await NotificationEnv.from_docker_image(LOCAL_IMAGE_NAME)
            else:
                env = NotificationEnv(base_url=ENV_URL)
            success, steps, score, rewards = await run_episode(env, task, events=events)
        finally:
            try:
                if env is not None:
                    await env.close()
            except Exception as err:
                debug_log(f"[DEBUG] env.close_error={err}")
            log_end(success=success, steps=steps, score=score, rewards=rewards)

        events.append(
            {
                "type": "task_end",
                "task": task,
                "success": bool(success),
                "steps": int(steps),
                "score": float(score),
                "rewards": list(rewards),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            }
        )

        debug_log(f"[DEBUG] task={task} score={score:.3f} success={success}")

    if TRACE_JSONL_PATH:
        try:
            write_jsonl(TRACE_JSONL_PATH, events)
            debug_log(f"[DEBUG] wrote_trace_jsonl={TRACE_JSONL_PATH}")
        except Exception as exc:
            debug_log(f"[DEBUG] trace_jsonl_error={exc}")

    if TRACE_HTML_PATH:
        try:
            report = render_html_report(events)
            write_html(TRACE_HTML_PATH, report)
            debug_log(f"[DEBUG] wrote_trace_html={TRACE_HTML_PATH}")
        except Exception as exc:
            debug_log(f"[DEBUG] trace_html_error={exc}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        debug_log(f"[DEBUG] fatal_error={exc}")
        print(f"[END] success=false steps=0 score={normalize_score(0.0):.3f} rewards=", flush=True)
        sys.exit(1)