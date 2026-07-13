---
title: "Building a Smart Document Inbox on Local AWS — Part 1: Foundations"
date: 2026-07-03 10:00:00 +0000
categories: [AWS, MiniStack]
tags: [aws, ministack, fastapi, boto3, docker, python, localstack, llm]
description: "Set up a free, fully local AWS environment with MiniStack and FastAPI — the foundation for building an LLM-powered Smart Document Inbox with zero cloud spend."
---

*MiniStack + FastAPI in 30 minutes.*

> **About this series.** Across this series, we are going to build one real and useful product — a **Smart Document Inbox**. The idea is simple: a user drops in a PDF or an article, and it is stored, understood by a local LLM (a summary, together with its topics and entities), made browsable, and finally emailed back to the user as an AI-generated summary. We build the whole thing on a laptop, with **no AWS account and no cloud spend**, using [MiniStack](https://ministack.org/) to emulate AWS and [Ollama](https://ollama.com) to run the LLM locally.
>
> Each part is sized for one focused weekend. This first part is the shortest one. Our goal here is to get the environment running and to make the first successful AWS call from a FastAPI application. By the end, we will have a skeleton that the remaining ten parts build upon.
{: .prompt-info }

## Why this series exists

Anyone who has tried to develop against AWS locally is probably familiar with the two less-than-ideal options:

1. **Develop against the real AWS.** This requires an account and credentials, and everything you do costs money. Worse, your unit tests now depend on the network and on a remote region whose behavior is not always predictable.
2. **Mock everything.** You stub out the `boto3` calls and hope that your mocks match the real behavior. In practice, the mocks tend to drift over time, so that the code passes the tests and then fails in production.

LocalStack used to provide a third option — a local AWS emulator — but its core services have since moved behind paid tiers. **MiniStack** is a free, MIT-licensed alternative: 60+ AWS services on a single port, roughly two-second startup, and around 30 MB of idle memory. Instead of pointing your AWS SDK at AWS, you point it at `http://localhost:4566`, and everything else stays the same.

This last sentence is the essential idea, and it is worth stating precisely, because it is exactly what makes the code we write this weekend also work in production.

## What we will cover in this part

- What MiniStack is, and the single idea that makes local AWS development portable
- Running MiniStack via `docker-compose`
- A FastAPI skeleton with config-driven, **endpoint-portable** AWS clients
- A `/healthz` route that proves the whole AWS path works
- The first real AWS call: creating and listing an S3 bucket, from code *and* from the AWS CLI

## Prerequisites

- **Docker** and **Docker Compose** (Docker Desktop, or Docker Engine together with the compose plugin)
- **Python 3.11+**
- The **AWS CLI** (`aws --version`), which is handy for poking at MiniStack from the terminal
- Basic familiarity with FastAPI (routes and the lifespan) and with the terminal

You do **not** need an AWS account, AWS credentials, or an internet connection (after the first image pull).

## The one idea: the endpoint switch

Every call made through an AWS SDK eventually reaches an HTTPS endpoint such as `https://s3.us-east-1.amazonaws.com`. `boto3` computes this URL for you from the service name and the region. Every client, however, accepts an explicit override:

```python
boto3.client("s3", endpoint_url="http://localhost:4566")
```

When `endpoint_url` is set, the SDK talks to MiniStack. When it is left unset, the very same code talks to the real AWS. It is also worth noting that MiniStack does not validate credentials; any non-empty values will work, and the common convention is to use `test` / `test`.

Accordingly, the design goal for this part — and the pattern we will reuse in every part that follows — is to **inject `endpoint_url` from configuration, and only when it is set**. Local development sets it; production leaves it empty. This is a single switch, with no code branches and no `if ENV == "local"` checks scattered across the codebase.

Let us build the skeleton around this idea.

## Build

### 1. Project layout

We will grow this repository across the series. The layout below is where we start; new folders will be added as we need them. The complete repository is available at [github.com/chuan2019/docinbox](https://github.com/chuan2019/docinbox).

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

To create it:

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

Within a second or two, MiniStack is listening on port 4566. We can verify this using its health endpoint:

```bash
curl http://localhost:4566/_ministack/health
```

**Example output** (the service list is abbreviated):

```
% curl http://localhost:4566/_ministack/health | jq .
{
  "services": {
    "account": "available",
    "acm": "available",
    "apigateway": "available",
    "cloudformation": "available",
    "dynamodb": "available",
    "dynamodbstreams": "available",
    "ec2": "available",
    "iam": "available",
    "kms": "available",
    "lambda": "available",
    "s3": "available",
    "secretsmanager": "available",
    "ses": "available",
    "sns": "available",
    "sqs": "available",
    "ssm": "available",
    "sts": "available",
    ... ...
    "bedrock": "available",
    "bedrock-runtime": "available",
    "kafka": "available"
  },
  "edition": "light",
  "version": "1.4.1",
  "ready_scripts": {
    "status": "completed",
    "total": 0,
    "completed": 0,
    "failed": 0
  }
}
```

You should receive a `200` response with a JSON body that reports the status of each service. This means the infrastructure is up. Now we can talk to it.

> Remember to add `data/` to your `.gitignore`; this directory holds local emulator state, not source code.
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

We are deliberately using **synchronous** `boto3` in these early parts, because it is the most familiar and because it keeps the focus on the AWS concepts themselves. We switch the hot paths to async (`aioboto3`) in **Part 3**, once there is real I/O to overlap.

### 4. Configuration

All environment-specific values are kept in a single, typed settings object. Nothing related to the AWS endpoints is hardcoded anywhere else.

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

`.env.example` (copy it to `.env` for local development):

```dotenv
AWS_ENDPOINT_URL=http://localhost:4566
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test
```

```bash
cp .env.example .env
```

The important line here is `AWS_ENDPOINT_URL=http://localhost:4566`. If we delete this single line, the same application points at the real AWS.

### 5. The portable client factory

This small module is the backbone of the entire series. Every part obtains its AWS clients from here, so that the endpoint switch is defined in exactly one place.

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

Two deliberate choices are worth pointing out here:

- **`endpoint_url` is added only when it is set.** This is the switch, and there are no environment conditionals anywhere else.
- **The clients are cached** (`lru_cache`). `boto3` clients are thread-safe and comparatively expensive to construct, so we create them once and reuse them. We will rely on this throughout the series.

### 6. The FastAPI app

Next we build a minimal application with a health check that actually exercises the AWS path — not merely whether the process is alive, but whether we can reach AWS and authenticate. For this we use the STS `GetCallerIdentity` call, which is the cheapest possible round trip that proves the wiring works end to end.

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

A few notes tie back to good practice:

- We **catch specific exceptions** (`ClientError`, `BotoCoreError`) rather than using a bare `except`. A failed AWS call then becomes a clean `503` response, instead of a stack trace sent to the client.
- `create_bucket` is **idempotent**; calling it twice is not an error. This matters more than it may appear. It is the seed of a pattern that we will rely on heavily once queues and Lambdas begin redelivering messages.
- The list endpoint returns `{"buckets": [...]}`, a **named object** rather than a bare array, so that we can add `count` or paging later without breaking existing clients.

Run it:

```bash
uvicorn app.main:app --reload --port 8000
```

### 7. The first AWS calls

With both MiniStack (`docker compose up -d`) and the application (`uvicorn ...`) running:

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

Now we confirm that the same resource is visible from the **AWS CLI**, pointed at MiniStack:

```bash
aws --endpoint-url http://localhost:4566 s3 ls
# 2026-07-04 10:00:00 inbox-uploads
```

This last step is the important observation: our FastAPI application and the standard AWS CLI are talking to the *same* emulated AWS. Anything we build against MiniStack can be inspected with ordinary AWS tooling.

> **Tip:** exporting the endpoint saves some typing during development:
> ```bash
> export AWS_ENDPOINT_URL=http://localhost:4566
> export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1
> aws s3 ls      # no --endpoint-url needed
> ```
{: .prompt-tip }

### 8. A Makefile

Finally, we collect the routine commands into a small Makefile, so that every part of the series shares the same set of commands and the daily workflow stays consistent from one weekend to the next.

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

The daily loop then becomes `make up`, `make run`, and `make down` when we are finished.

## The general pattern

> **Portable takeaway (independent of this particular product):** any code that talks to AWS should obtain its clients from a **single factory that injects `endpoint_url` from configuration, and only when it is set**. This one decision gives us local development with no cloud account, deterministic integration tests (Part 10), and identical code in production. We should never scatter `if local:` checks or hardcode endpoints; instead, we centralize the switch, and the rest of the codebase never needs to know whether it is talking to an emulator or to the real service.
{: .prompt-tip }

## In production

> - **Drop the endpoint.** In the real AWS we leave `AWS_ENDPOINT_URL` unset, and the factory then targets the real service endpoints automatically.
> - **Drop the static credentials.** We do not ship `test`/`test`, and we do not ship *any* long-lived keys. Production workloads obtain credentials from the ambient chain: an **IAM role** (EKS Pod Identity or ECS Task Role) or SSO. `boto3` picks these up with no code change, as long as we simply do not pass `aws_access_key_id` / `aws_secret_access_key`.
> - **Health checks.** `GetCallerIdentity` is a reasonable reachability probe, but in a real deployment the `/healthz` (liveness) endpoint should stay cheap and should *not* depend on downstream services, while a separate `/readyz` (readiness) endpoint checks the dependencies. We keep a single endpoint here for simplicity and split them properly later.
{: .prompt-info }

## MiniStack notes and fidelity

- **Startup and footprint.** MiniStack starts in roughly two seconds and idles at around 30 MB, so it is cheap to run all day and cheap to spin up for each test run (we take advantage of this in Part 10). The image is about 270 MB on the first pull.
- **Useful environment variables (for this part):** `GATEWAY_PORT` (default `4566`), `LOG_LEVEL` (set it to `DEBUG` when something is mysterious), and `PERSIST_STATE` / `S3_PERSIST` (which persist the metadata and the object bytes across restarts). We turned persistence *on* for development, and we turn it *off* for CI determinism in Part 10.
- **The account id is fake.** `GetCallerIdentity` returns a placeholder account (`000000000000`). This is expected, since MiniStack is not authenticating you against a real account.
- **Credentials are not enforced.** Any credentials work, and, importantly, MiniStack generally does **not** enforce IAM permissions. This is convenient for velocity, but it also means that "it worked locally" does not prove that your IAM policy is correct. We address this directly in Part 8 (STS/IAM).

## Troubleshooting

- **`Could not connect to the endpoint URL`.** *Likely cause:* MiniStack is not running, or the port is wrong. *Fix:* run `make up`; check `docker compose ps`; confirm port 4566.
- **`/healthz` returns 503.** *Likely cause:* the application cannot reach MiniStack. *Fix:* check whether `AWS_ENDPOINT_URL` is set in `.env`, and whether the container is up.
- **The CLI cannot see a bucket that your app created.** *Likely cause:* the CLI is pointed at the real AWS. *Fix:* add `--endpoint-url http://localhost:4566` (or export it).
- **Buckets vanish after `docker compose down`.** *Likely cause:* persistence is off, or the volumes are not mounted. *Fix:* set `PERSIST_STATE=1` and `S3_PERSIST=1`, and mount the volumes.

## Weekend exercises

1. **Add a second service call.** Extend `/healthz` (or add a `/whoami` route) so that it also reports the STS `Arn`, not just the account. Notice that you did not need to touch the client factory to do this.
2. **Break it on purpose.** Stop MiniStack (`make down`) and hit `/healthz`. Read the exact exception. Then point `AWS_ENDPOINT_URL` at a wrong port and compare the error. Knowing what these failures *look like* will save you an hour later.
3. **Round-trip an object.** Using only the AWS CLI against MiniStack, run `aws s3 cp` to copy a local file into `inbox-uploads`, and then `aws s3 ls s3://inbox-uploads`. This previews Part 3, where the application does the same thing.
4. **Prove portability (a thought experiment plus a diff).** Comment out `AWS_ENDPOINT_URL` in `.env`. What happens when you hit `/healthz`, and *why*? (You will get a credentials or region error from the real AWS, which is exactly the point: the code did not change, only the configuration did.)

## What is next

At this point we have a running local AWS and an application that can talk to it portably. For now, however, our configuration is a `.env` file, and there are essentially no secrets involved. Real applications pull their configuration and secrets from AWS itself.

**[Part 2 — Configuration & Secrets: SSM Parameter Store + Secrets Manager](/posts/smart-document-inbox-part-2-configuration-and-secrets/).** We will move `docinbox`'s settings (the bucket names, the LLM model name, and the feature flags) into SSM Parameter Store, store a signing key in Secrets Manager, load them at startup into the same typed settings object, and meet STS properly. The endpoint switch stays the same; we simply add two new services.

The source code used in this part is available in the companion GitHub repository: [github.com/chuan2019/docinbox](https://github.com/chuan2019/docinbox) (tag `part-01`).

*See you next weekend.*
