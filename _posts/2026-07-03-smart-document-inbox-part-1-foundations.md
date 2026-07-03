---
title: "Building a Smart Document Inbox on Local AWS — Part 1: Foundations"
date: 2026-07-03 10:00:00 +0000
categories: [AWS, MiniStack]
tags: [aws, ministack, fastapi, boto3, docker, python, localstack, llm]
description: "Set up a free, fully local AWS environment with MiniStack and FastAPI — the foundation for building an LLM-powered Smart Document Inbox with zero cloud spend."
---

*MiniStack + FastAPI in 30 minutes.*

> **This series.** We're building one real, useful product across the whole series — a **Smart Document Inbox**: drop in a PDF or article, and it gets stored, understood by a local LLM (summary, topics, entities), made browsable, and emailed back to you as an AI summary. We build it entirely on your laptop, with **zero AWS account and zero cloud spend**, using [MiniStack](https://ministack.org/) to emulate AWS and [Ollama](https://ollama.com) to run the LLM.
>
> Every part is sized for one focused weekend. This first one is the shortest: we get the environment running and make our first successful AWS call from FastAPI. By the end you'll have a skeleton that the next ten parts build on.
{: .prompt-info }

## Why this series exists

If you've tried to develop against AWS locally, you know the two bad options:

1. **Develop against real AWS.** You need an account, credentials, and you pay for everything. Worse, your "unit tests" now depend on the network and a us-east-1 that occasionally has opinions about your Tuesday.
2. **Mock everything.** You stub out `boto3` calls and hope your mocks match reality. They drift. The code passes tests and fails in prod.

LocalStack used to be the third option — a local AWS emulator — but its core services moved behind paid tiers. **MiniStack** is the free, MIT-licensed answer: 60+ AWS services on a single port, ~2-second startup, ~30 MB idle memory. You point your AWS SDK at `http://localhost:4566` instead of AWS, and everything else stays the same.

That last sentence is the entire trick, and it's worth saying precisely, because it's what makes the code you write this weekend also work in production.

## What you'll learn this part

- What MiniStack is and the one idea that makes local AWS development portable
- Running MiniStack via `docker-compose`
- A FastAPI skeleton with config-driven, **endpoint-portable** AWS clients
- A `/healthz` route that proves the whole AWS path works
- Your first real AWS call: create and list an S3 bucket — from code *and* the AWS CLI

## Prerequisites

- **Docker** + **Docker Compose** (Docker Desktop, or Docker Engine + the compose plugin)
- **Python 3.11+**
- The **AWS CLI** (`aws --version`) — handy for poking at MiniStack from the terminal
- Basic familiarity with FastAPI (routes, the lifespan) and the terminal

You do **not** need an AWS account, AWS credentials, or an internet connection (after the first image pull).

## The one idea: the endpoint switch

Every AWS SDK call ultimately hits an HTTPS endpoint like `https://s3.us-east-1.amazonaws.com`. `boto3` computes that URL for you from the service name and region. But every client accepts an override:

```python
boto3.client("s3", endpoint_url="http://localhost:4566")
```

Set `endpoint_url`, and the SDK talks to MiniStack. Leave it unset, and the exact same code talks to real AWS. MiniStack also doesn't validate credentials — any non-empty values work (the convention is `test` / `test`).

So the design goal for this part — and the pattern we'll reuse in every part after — is: **inject `endpoint_url` from configuration, and only when it's set.** Local dev sets it; production leaves it empty. One switch, no code branches, no `if ENV == "local"` scattered through the codebase.

Let's build the skeleton around that idea.

## Build

### 1. Project layout

We'll grow this repo across the series. Here's where we start; new folders arrive as we need them.

```
docinbox/                 # the Smart Document Inbox
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app + routes
│   ├── config.py         # pydantic-settings config
│   └── aws/
│       ├── __init__.py
│       └── clients.py    # the portable AWS client factory
├── docker-compose.yml    # ministack (more services later)
├── Makefile              # make up / down / seed / test
├── requirements.txt
└── .env.example
```

Create it:

```bash
mkdir -p docinbox/app/aws
cd docinbox
```

### 2. Run MiniStack

`docker-compose.yml`:

```yaml
services:
  ministack:
    image: ministackorg/ministack:latest
    ports:
      - "4566:4566"
    environment:
      - LOG_LEVEL=INFO
      # Keep created resources across restarts during development.
      # (In CI you'll want these OFF for a clean slate — that's Part 10.)
      - PERSIST_STATE=1
      - S3_PERSIST=1
    volumes:
      - ./data/state:/tmp/ministack-state
      - ./data/s3:/tmp/ministack-data/s3
```

Start it:

```bash
docker compose up -d
```

In a second or two, MiniStack is listening on port 4566. Verify with its health endpoint:

```bash
curl http://localhost:4566/_ministack/health
```

You should get a `200` with a JSON body reporting service status. That's your infrastructure up. Now let's talk to it.

> Add `data/` to your `.gitignore` — that's local emulator state, not source.
{: .prompt-tip }

### 3. Python environment

`requirements.txt`:

```
fastapi>=0.111
uvicorn[standard]>=0.30
boto3>=1.34
pydantic-settings>=2.2
httpx>=0.27
```

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

We're using **synchronous** `boto3` in these early parts because it's the most familiar and keeps the focus on AWS concepts. We switch the hot paths to async (`aioboto3`) in **Part 3**, once there's real I/O to overlap.

### 4. Configuration

All environment-specific values live in one typed settings object. Nothing about AWS endpoints is hardcoded anywhere else.

`app/config.py`:

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, read from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Set to http://localhost:4566 locally; LEAVE UNSET in real AWS.
    aws_endpoint_url: str | None = None
    aws_region: str = "us-east-1"

    # MiniStack ignores these; real AWS uses the ambient credential chain
    # (IAM role, SSO, etc.) and ignores these defaults.
    aws_access_key_id: str = "test"
    aws_secret_access_key: str = "test"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so we parse the environment once."""
    return Settings()
```

`.env.example` (copy to `.env` for local dev):

```dotenv
AWS_ENDPOINT_URL=http://localhost:4566
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test
```

```bash
cp .env.example .env
```

The important line is `AWS_ENDPOINT_URL=http://localhost:4566`. Delete that one line and the same app points at real AWS.

### 5. The portable client factory

This tiny module is the backbone of the entire series. Every part gets its AWS clients from here, so the endpoint switch is defined in exactly one place.

`app/aws/clients.py`:

```python
from functools import lru_cache
from typing import Any

import boto3

from app.config import get_settings


def _client_kwargs() -> dict[str, Any]:
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "region_name": settings.aws_region,
        "aws_access_key_id": settings.aws_access_key_id,
        "aws_secret_access_key": settings.aws_secret_access_key,
    }
    # Only override the endpoint when configured (i.e. local dev).
    # In production this stays unset and boto3 targets real AWS.
    if settings.aws_endpoint_url:
        kwargs["endpoint_url"] = settings.aws_endpoint_url
    return kwargs


@lru_cache
def get_client(service: str) -> Any:
    """Return a cached boto3 client for `service`, wired to MiniStack or AWS.

    boto3 clients are thread-safe, so caching one per service is both safe
    and the recommended pattern (client creation is relatively expensive).
    """
    return boto3.client(service, **_client_kwargs())
```

Two deliberate choices worth calling out:

- **`endpoint_url` is only added when set.** That's the switch. No environment conditionals anywhere else.
- **Clients are cached** (`lru_cache`). `boto3` clients are thread-safe and comparatively expensive to construct; you make them once and reuse them. We'll lean on this everywhere.

### 6. The FastAPI app

Now a minimal app with a health check that actually exercises the AWS path — not just "is the process alive," but "can I reach AWS and authenticate." We use STS `GetCallerIdentity` for that: it's the cheapest possible round-trip that proves the wiring end to end.

`app/main.py`:

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from botocore.exceptions import BotoCoreError, ClientError

from app.aws.clients import get_client

app = FastAPI(title="Smart Document Inbox")


@app.get("/healthz")
def healthz() -> JSONResponse:
    """Liveness + AWS reachability.

    Returns 200 only if we can make a real (emulated) AWS call.
    """
    try:
        identity = get_client("sts").get_caller_identity()
    except (ClientError, BotoCoreError) as exc:
        # Reaching AWS/MiniStack failed — report unhealthy, don't crash.
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(exc)},
        )
    return JSONResponse(
        content={"status": "ok", "account": identity["Account"]}
    )


@app.post("/buckets/{name}")
def create_bucket(name: str) -> dict[str, str]:
    """Create an S3 bucket — our first real write to (emulated) AWS."""
    s3 = get_client("s3")
    try:
        s3.create_bucket(Bucket=name)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass  # idempotent: fine if it already exists
    return {"created": name}


@app.get("/buckets")
def list_buckets() -> dict[str, list[str]]:
    s3 = get_client("s3")
    resp = s3.list_buckets()
    return {"buckets": [b["Name"] for b in resp["Buckets"]]}
```

A few notes tying back to good practice:

- We **catch specific exceptions** (`ClientError`, `BotoCoreError`), never a bare `except`. A failed AWS call turns into a clean `503`, not a stack trace to the client.
- `create_bucket` is **idempotent** — calling it twice is not an error. This matters more than it looks; it's the seed of a pattern we rely on heavily once queues and Lambdas start redelivering messages.
- The list endpoint returns `{"buckets": [...]}`, a **named object**, not a bare array — so we can add `count` or paging later without breaking clients.

Run it:

```bash
uvicorn app.main:app --reload --port 8000
```

### 7. Your first AWS calls

With both MiniStack (`docker compose up -d`) and the app (`uvicorn ...`) running:

```bash
# Health check — proves the AWS path works end to end
curl -s localhost:8000/healthz
# {"status":"ok","account":"000000000000"}

# Create a bucket via your API
curl -s -X POST localhost:8000/buckets/inbox-uploads
# {"created":"inbox-uploads"}

# List buckets via your API
curl -s localhost:8000/buckets
# {"buckets":["inbox-uploads"]}
```

Now confirm the same resource is visible from the **AWS CLI**, pointed at MiniStack:

```bash
aws --endpoint-url http://localhost:4566 s3 ls
# 2026-07-04 10:00:00 inbox-uploads
```

That last step is the "aha": your FastAPI app and the standard AWS CLI are talking to the *same* emulated AWS. Anything you build against MiniStack, you can inspect with ordinary AWS tooling.

> **Tip:** exporting the endpoint saves typing during development:
>
> ```bash
> export AWS_ENDPOINT_URL=http://localhost:4566
> export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1
> aws s3 ls      # no --endpoint-url needed
> ```

### 8. A Makefile

So every part has the same muscle memory:

`Makefile`:

```makefile
.PHONY: up down logs run health

up:            ## Start MiniStack
	docker compose up -d

down:          ## Stop MiniStack
	docker compose down

logs:          ## Tail MiniStack logs
	docker compose logs -f ministack

run:           ## Run the FastAPI app
	uvicorn app.main:app --reload --port 8000

health:        ## Check MiniStack + app health
	curl -s localhost:4566/_ministack/health && echo && curl -s localhost:8000/healthz
```

Now the daily loop is `make up`, `make run`, and `make down` when you're finished.

## The general pattern

> **Portable takeaway (independent of this product):** Any code that talks to AWS should get its clients from a **single factory that injects `endpoint_url` from configuration, only when set**. This one decision buys you: local development with no cloud account, deterministic integration tests (Part 10), and identical code in production. Never scatter `if local:` checks or hardcode endpoints — centralize the switch, and the rest of your codebase never knows whether it's talking to an emulator or the real thing.
{: .prompt-tip }

## In production

> - **Drop the endpoint.** In real AWS you leave `AWS_ENDPOINT_URL` unset; the factory then targets real service endpoints automatically.
> - **Drop the static credentials.** Don't ship `test`/`test` — and don't ship *any* long-lived keys. Production workloads get credentials from the ambient chain: an **IAM role** (EKS Pod Identity / ECS Task Role) or SSO. `boto3` picks these up with no code change; just don't pass `aws_access_key_id` / `aws_secret_access_key`.
> - **Health checks.** `GetCallerIdentity` is a fine reachability probe, but in a real deployment your `/healthz` (liveness) should stay cheap and *not* depend on downstream services, while a separate `/readyz` (readiness) checks dependencies. We keep one endpoint here for simplicity and split them properly later.
{: .prompt-info }

## MiniStack notes & fidelity

- **Startup & footprint.** MiniStack starts in ~2 s and idles around ~30 MB, so it's cheap to run all day and cheap to spin up per test run (we exploit that in Part 10). The image is ~270 MB on first pull.
- **Useful env vars (this part):** `GATEWAY_PORT` (default `4566`), `LOG_LEVEL` (`DEBUG` when something's mysterious), `PERSIST_STATE` / `S3_PERSIST` (persist metadata / object bytes across restarts). We turned persistence *on* for development; we turn it *off* for CI determinism in Part 10.
- **The account id is fake.** `GetCallerIdentity` returns a placeholder account (`000000000000`). That's expected — MiniStack isn't authenticating you against a real account.
- **Credentials aren't enforced.** Any credentials work, and (importantly) MiniStack generally does **not** enforce IAM permissions. That's great for velocity but means "it worked locally" doesn't prove your IAM policy is correct. We confront this directly in Part 8 (STS/IAM).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Could not connect to the endpoint URL` | MiniStack not running, or wrong port | `make up`; check `docker compose ps`; confirm port 4566 |
| `/healthz` returns 503 | App can't reach MiniStack | Is `AWS_ENDPOINT_URL` set in `.env`? Is the container up? |
| CLI can't see a bucket your app created | CLI pointed at real AWS | Add `--endpoint-url http://localhost:4566` (or export it) |
| Buckets vanish after `docker compose down` | Persistence off, or volumes not mounted | Set `PERSIST_STATE=1` + `S3_PERSIST=1` and mount the volumes |

## Weekend exercises

1. **Add a second service call.** Extend `/healthz` (or add `/whoami`) to also report the STS `Arn`, not just the account. Notice you didn't touch the client factory to do it.
2. **Break it on purpose.** Stop MiniStack (`make down`) and hit `/healthz`. Read the exact exception. Then point `AWS_ENDPOINT_URL` at a wrong port and compare the error. Knowing what these failures *look like* saves you an hour later.
3. **Round-trip an object.** Using only the AWS CLI against MiniStack, `aws s3 cp` a local file into `inbox-uploads`, then `aws s3 ls s3://inbox-uploads`. This previews Part 3, where the app does it.
4. **Prove portability (thought experiment + diff).** Comment out `AWS_ENDPOINT_URL` in `.env`. What happens when you hit `/healthz`, and *why*? (You'll get a credentials/region error from real AWS — which is exactly the point: the code didn't change, only the config did.)

## What's next

We have a running local AWS and an app that can talk to it portably. But right now our config is a `.env` file and there are no secrets to speak of. Real apps pull configuration and secrets from AWS itself.

**Part 2 — Configuration & Secrets: SSM Parameter Store + Secrets Manager.** We'll move `docinbox`'s settings (bucket names, the LLM model name, feature flags) into SSM Parameter Store, store a signing key in Secrets Manager, load them at startup into the same typed settings object, and meet STS properly. Same endpoint switch, two new services.

*See you next weekend.*
