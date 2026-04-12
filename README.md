---
title: "AI Notification Gatekeeper"
sdk: docker
colorFrom: blue
colorTo: green
---

# AI Notification Gatekeeper

An OpenEnv benchmark where an agent decides how to handle smartphone notifications under changing user context.

## Environment Description And Motivation

Modern users receive mixed notifications: urgent work pings, transactional OTPs, social chats, and adversarial spam. A useful notification assistant must do more than classify message type; it must reason over context (user state, active tasks, sender trust, urgency cues) and choose a suitable action.

This environment is designed to benchmark that behavior using scenario-driven episodes across three difficulty tiers.

Core goals:
- Measure context-aware decision quality, not just keyword matching.
- Test robustness to adversarial/fake urgency patterns.
- Provide a reproducible benchmark for RL, rule-based, and LLM policies.

## Action Space

Discrete action space with 4 actions:

- `notify_now`: show immediately.
- `silent`: suppress notification.
- `delay`: defer for later.
- `escalate`: high-priority interrupt.

## Observation Space

Each step returns a structured observation with the following fields:

- `app: str`
- `category: str`
- `sender_type: str`
- `urgency_hint: float` in `[0, 1]`
- `message_frequency: int`
- `content_keywords: list[str]`
- `user_state: str`
- `time_of_day: str`
- `active_tasks: list[str]`
- `sender_history: str`
- `sender_trust: float` in `[0, 1]`
- `step_number: int`
- `task: str`
- `feedback: str`
- `reward: float | None`
- `done: bool`

## Reward And Episode Scoring

Per-step reward:
- `1.0` for expected action
- `0.5` for acceptable but non-optimal action
- `0.0` for wrong action

Episode score is average reward over actual episode steps:

`episode_score = total_reward / num_steps`

## Tasks And Expected Difficulty

- `signal_clarity` (easy): high-signal cases (clear urgent, clear spam, clear transactional).
- `context_aware` (medium): same app/category can require different actions based on user context.
- `adversarial_signals` (hard): fake urgency, scam patterns, and spam-to-real-emergency transitions.

Current scenario counts:
- `signal_clarity`: 25
- `context_aware`: 25
- `adversarial_signals`: 25

## Setup

### 1. Create environment and install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2. Run server locally

```bash
uvicorn notify_env.server.app:app --host 0.0.0.0 --port 7860 --reload
```

### 3. Quick API check

```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task":"signal_clarity"}'
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"decision":"notify_now"}'
```

## Usage

### OpenAI-compatible runner

`inference.py` runs policy inference through an OpenAI-compatible endpoint.

Required env vars:
- `API_KEY`
- `API_BASE_URL` (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME` (default: `Qwen/Qwen2.5-72B-Instruct`)
- `NOTIF_ENV_URL` (local or deployed environment URL)

Run:

```bash
python inference.py
```

### Ollama runner

`ollamainference.py` runs local model inference through Ollama.

Common vars:
- `OLLAMA_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `qwen2.5:7b`)
- `NOTIF_ENV_URL`
- `NOTIF_TASK` (optional: run a single task)

Run:

```bash
python ollamainference.py
```

Both runners emit structured logs:
- `[START]`
- `[STEP]`
- `[END]`

## Baseline Scores

Reference deterministic-policy baselines (computed on current 25x3 scenario set):

| Policy | signal_clarity | context_aware | adversarial_signals |
|---|---:|---:|---:|
| always_silent | 0.48 | 0.22 | 0.44 |
| always_delay | 0.12 | 0.34 | 0.34 |
| always_notify_now | 0.42 | 0.58 | 0.18 |
| always_escalate | 0.14 | 0.08 | 0.24 |

Observed model-policy baseline from `ollamainference.py` (`qwen2.5:7b`, 25-step episodes):

| Runner | signal_clarity | context_aware | adversarial_signals | mean |
|---|---:|---:|---:|---:|
| ollamainference.py (`qwen2.5:7b`) | 0.84 | 0.58 | 0.64 | 0.687 |

Notes:
- `warmup_call_error=timed out` can happen if Ollama takes too long to answer the short warmup request.
- This does not invalidate the run when the task loops execute and emit complete `[START]`/`[STEP]`/`[END]` blocks.

Interpretation:
- `signal_clarity` rewards suppression of obvious spam and escalation of clear emergencies.
- `context_aware` favors timely contextual actions (`notify_now` performs relatively better here).
- `adversarial_signals` penalizes naive always-on escalation/notification behavior.
