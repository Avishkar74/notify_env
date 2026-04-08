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
uvicorn notify_env.server.app:app --host 0.0.0.0 --port 8000 --reload
```

## Quick API Check

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task":"signal_clarity"}'
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" -d '{"decision":"notify_now"}'
```

## Baseline Script

`inference.py` is included at project root and emits:

- `[START]`
- `[STEP]`
- `[END]`

Configure environment variables before running:

- `HF_TOKEN`
- `MODEL_NAME`
- `API_BASE_URL`
- `NOTIF_ENV_URL`

Run:

```bash
python inference.py
```
