---
title: "AI Notification Gatekeeper"
sdk: docker
colorFrom: blue
colorTo: green
---

# AI Notification Gatekeeper

OpenEnv environment for context-aware notification decisions across three tasks.

## Tasks

- `signal_clarity`: obvious signals (boss urgent, promo spam, OTP)
- `context_aware`: same notification can require different actions by context
- `adversarial_signals`: fake urgency, spam-then-real emergencies, escalation chains

## Action Space

- `notify_now`
- `silent`
- `delay`
- `escalate`

## Observation Fields

- `app`, `category`, `sender_type`, `urgency_hint`, `message_frequency`, `content_keywords`
- `user_state`, `active_tasks`, `sender_history`, `sender_trust`
- `step_number`, `task`, `feedback`

## Reward

Per-step:
- `1.0` correct action
- `0.5` acceptable action
- `0.0` wrong action

Episode score:

`score = total_reward / 5.0`

## Local Run

```bash
pip install -e .
uvicorn notify_env.server.app:app --host 0.0.0.0 --port 7860 --reload
```

## Quick API Check

```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task":"signal_clarity"}'
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"decision":"notify_now"}'
```

## Baseline Script

`inference.py` is included at project root and emits:

- `[START]`
- `[STEP]`
- `[END]`

Configure environment variables before running:

- `OPENAI_API_KEY` (preferred)
- `HF_TOKEN` (fallback)
- `MODEL_NAME`
- `API_BASE_URL`
- `NOTIF_ENV_URL`

Run:

```bash
python inference.py
```

## Ollama Test Runner

`ollamainference.py` is included for local inference-flow validation without OpenAI credentials.

Defaults:
- `OLLAMA_URL=http://localhost:11434`
- `OLLAMA_MODEL=qwen2.5:7b`

Run with one task:

```bash
OLLAMA_URL=http://localhost:11434 \
OLLAMA_MODEL=qwen2.5:7b \
NOTIF_ENV_URL=https://Avishkar-00-notify-env.hf.space \
NOTIF_TASK=signal_clarity \
python ollamainference.py
```

Notes:
- This validates prompting, action parsing, episode loop, and `[START]/[STEP]/[END]` logging.
- It does not replace final OpenAI baseline evidence for submission scoring.

## Baseline Results Snapshot

| Runner | Task | Score | Status |
|---|---|---:|---|
| `ollamainference.py` (`qwen2.5:7b`) | `signal_clarity` | `0.700` | Verified |
| `inference.py` (OpenAI credentials) | all 3 tasks | pending | Awaiting API key |

## Reinforcement Learning (RL) Purpose

This environment is designed to evaluate simple reinforcement learning (RL) and decision-making behaviors for a notification-handling agent. Each episode consists of a short sequence of steps (5 steps) where the agent observes notification and user-context features and selects one action from the discrete action space (`notify_now`, `silent`, `delay`, `escalate`).

- The reward function is episodic and sparse: correct actions receive positive reward, partially-correct actions receive smaller reward, and wrong actions receive zero. Episodes are scored by average per-step reward.
- This setup is suitable for training or evaluating policy-based agents, imitation learning baselines, or RL algorithms that must learn context-conditioned decision policies under partial observability.
- For this submission baseline we use the LLM as a policy (prompt → action) and measure performance via the episode score reported by `inference.py`.

Practical uses:
- Rapid prototyping of policies for notification triage.
- Measuring policy robustness across adversarial or context-shift scenarios.
- Comparing LLM-based policies against learned RL policies in a reproducible benchmark.
