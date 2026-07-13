---
title: "Building a Smart Document Inbox on Local AWS — Part 2: Configuration & Secrets"
date: 2026-07-12 10:00:00 +0000
categories: [AWS, MiniStack]
tags: [aws, ministack, fastapi, boto3, python, ssm, parameter-store, secrets-manager, sts, pydantic]
description: "Move app configuration into SSM Parameter Store and secrets into Secrets Manager — loaded at startup, cached with a TTL, and kept out of the logs — all locally with MiniStack."
---

*SSM Parameter Store + Secrets Manager, plus a first look at STS.*

> **About this series.** Across this series, we are going to build one real and useful product — a **Smart Document Inbox**. The idea is simple: a user drops in a PDF or an article, and it is stored, understood by a local LLM (a summary, together with its topics and entities), made browsable, and finally emailed back to the user as an AI-generated summary. We build the whole thing on a laptop, with **no AWS account and no cloud spend**, using [MiniStack](https://ministack.org/) to emulate AWS and [Ollama](https://ollama.com) to run the LLM locally.
>
> **Catching up?** In [Part 1](/posts/smart-document-inbox-part-1-foundations/), we set up MiniStack via `docker-compose`, created a FastAPI skeleton, and built the **portable client factory** — the module that injects `endpoint_url` from configuration, so that the same code talks to MiniStack locally and to the real AWS in production. This part assumes that skeleton is in place.
{: .prompt-info }

## Where Part 1 left us

At the end of Part 1, the application runs, talks to MiniStack, and can create S3 buckets. All of its configuration lives in a `.env` file. For the two values we have so far (the endpoint URL and the region), this is appropriate: they are *bootstrap* settings, and they belong in the environment. However, the application is about to grow real application-level configuration — which bucket to store documents in, which LLM model to run, and whether the email digest feature is enabled — and it will also need its first secret, a signing key.

Keeping all of these values in `.env` files causes some well-known problems:

1. **Configuration drift.** Every developer, container, and worker carries its own copy of the values, and the copies drift apart over time.
2. **Secrets in plaintext files.** A secret stored in `.env` also ends up in the shell history, and sooner or later in the git history.
3. **No single source of truth.** When the bucket name changes, it needs to be updated in several places, and it is easy to miss one.

The AWS-native answer is to treat configuration and secrets as services in their own right: **SSM Parameter Store** for configuration, and **Secrets Manager** for secrets. The application then fetches both at startup from one authoritative place. Since MiniStack emulates both services, we can build the exact production code path locally.

In this post, I am going to demonstrate how to move `docinbox`'s configuration into SSM Parameter Store, store a signing key in Secrets Manager, and load both at startup into a typed configuration object.

## What we will cover in this part

- **SSM Parameter Store**: `String` vs `SecureString`, hierarchical paths (`/docinbox/dev/...`), and fetching a whole configuration tree in one call
- **Secrets Manager**: what it adds over SSM, and when it is worth paying for in production
- The startup-load + TTL-cache pattern for configuration
- An idempotent seed script (`bootstrap/`), a convention that every later part reuses
- Keeping secrets out of the logs with pydantic's `SecretStr`
- STS `GetCallerIdentity`, which Part 8 builds upon

## Prerequisites

- The Part 1 skeleton, running (`make up`, `make run`, and `/healthz` returns `ok`)
- Nothing new to install; SSM, Secrets Manager, and STS are already included in the MiniStack container we are running

## Concepts

### SSM Parameter Store vs Secrets Manager

**SSM Parameter Store** is a hierarchical key-value store for configuration. A parameter has a name (by convention a path, such as `/docinbox/dev/s3/bucket-name`), a type, and a value:

- **`String`** — plain configuration, such as bucket names, model names, and feature flags.
- **`StringList`** — comma-separated values, for the cases where a list is needed.
- **`SecureString`** — encrypted at rest with a KMS key, and decrypted on read when requested (`WithDecryption=True`). This is a reasonable home for low-ceremony secrets.

**Secrets Manager** stores only secrets. Compared to a `SecureString` parameter, it adds:

- **Rotation** — it can invoke a Lambda function on a schedule to rotate the secret (database passwords are the classic case).
- **Versioning with staging labels** — `AWSCURRENT` / `AWSPREVIOUS`, so that a rotation can be rolled back.
- **Structured secrets** — the value is commonly a JSON blob (for example `{"username": ..., "password": ...}`), fetched as one unit.
- **Cross-account access and replication** — a secret can be shared across accounts or regions with resource policies.

For `docinbox`, we will follow this rule:

- The **bucket name** and queue URLs are not secret, so they go into SSM as `String` parameters.
- The **LLM model name** also goes into SSM; it is configuration that we will want to change without redeploying (Part 7 makes use of this).
- **Feature flags** go into SSM as `String` parameters holding `"true"` or `"false"`.
- The **signing key**, and later any API keys or database passwords, go into **Secrets Manager**, where rotation, versioning, and auditing are available.

> **In production:** cost is part of this decision. SSM standard-tier parameters are free (up to 10,000 per account), while Secrets Manager charges about $0.40 per secret per month, plus a per-10k-API-calls fee. This is why teams generally put ordinary configuration into SSM and reserve Secrets Manager for values that actually rotate. Locally both are free, but the habit is worth building now.
{: .prompt-info }

### Hierarchical parameter paths

Naming the parameters `/docinbox/dev/s3/bucket-name` instead of `DOCINBOX_BUCKET` gives us a real API feature: `GetParametersByPath` fetches an entire subtree in one paginated call.

```
/docinbox/dev/s3/bucket-name        → inbox-uploads
/docinbox/dev/llm/model-name        → llama3.2:3b
/docinbox/dev/features/email-digest → false
```

One call at startup returns the application's whole configuration tree. The environment becomes a path segment (`/docinbox/dev/...`, `/docinbox/prod/...`), so deploying to a different environment changes one prefix rather than many variable names. In the real AWS, IAM policies can also grant access by path prefix; the worker role in Part 8 could be allowed to read `/docinbox/prod/*` and nothing else.

### Fetching and caching configuration

There are two common ways to get configuration fetching wrong. The first is to call `get_parameter` inside a request handler, which adds an SSM round trip to every request; in the real AWS it also leads to throttling (the SSM throughput limits are shared account-wide) and to unnecessary cost. The second is to fetch the configuration once at import time and never again, which means that changing a feature flag requires restarting every instance.

The approach we take in this part sits in between: load the configuration at startup in the FastAPI lifespan, cache it in memory, and re-fetch it when the cache is older than a TTL. With a five-minute TTL, configuration changes propagate within five minutes, and SSM sees one call per instance per five minutes. AWS's own Parameters and Secrets Lambda Extension implements the same idea as a sidecar; here we build it by hand, so that it is not magic.

## Build

The target layout after this part is shown below (new files are marked):

```
docinbox/
├── app/
│   ├── main.py               # + lifespan, /whoami, /config
│   ├── config.py             # bootstrap settings (env) — slightly extended
│   ├── app_config.py         # NEW: typed app config from SSM + Secrets Manager
│   └── aws/
│       └── clients.py        # unchanged
├── bootstrap/
│   ├── __init__.py           # NEW
│   └── seed.py               # NEW: idempotent resource seeding
├── docker-compose.yml        # unchanged
├── Makefile                  # + make seed
└── ...
```

Note that `docker-compose.yml` and `aws/clients.py` do not change. SSM, Secrets Manager, and STS are already listening on port 4566, and the client factory from Part 1 works for them as-is: `get_client("ssm")` is all we need.

### 1. Separating bootstrap settings from app config

The distinction that organizes this part is the following:

- **Bootstrap settings** answer the question "where is AWS, and which environment am I in?" — the endpoint, the region, and the environment name. These have to come from the environment, because we cannot fetch configuration from AWS before we know where AWS is. This is our existing `Settings` class.
- **App config** answers the question "how should the application behave?" — bucket names, model names, flags, and secrets. From now on, these come from AWS, through a new typed object.

We extend `app/config.py` with the new bootstrap values:

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Bootstrap configuration, read from environment / .env.

    Only what we need BEFORE we can talk to AWS. Everything else
    (bucket names, model names, flags) lives in SSM — see app_config.py.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Set to http://localhost:4566 locally; LEAVE UNSET in real AWS.
    aws_endpoint_url: str | None = None
    aws_region: str = "us-east-1"

    aws_access_key_id: str = "test"
    aws_secret_access_key: str = "test"

    # Which config tree to read: /docinbox/<app_env>/...
    app_env: str = "dev"
    # How long loaded app config stays fresh before we re-fetch.
    config_ttl_seconds: int = 300
    # Exposes GET /config. Never enable in production.
    debug_routes: bool = True


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so we parse the environment once."""
    return Settings()
```

We also add two lines to `.env.example` (and copy them into `.env`):

```dotenv
APP_ENV=dev
DEBUG_ROUTES=true
```

### 2. Seeding the parameters and the secret

Every part from here on needs certain resources to exist before the application starts: parameters now, and buckets, tables, and queues later. The convention we set up once and reuse for the rest of the series is an **idempotent** `bootstrap/seed.py`, which can be run any number of times and converges the emulator to the state the application expects. A fresh clone then becomes `make up && make seed`.

`bootstrap/seed.py`:

```python
"""Idempotent seeding for docinbox resources.

Run with:  python -m bootstrap.seed
Safe to run repeatedly — it converges, it doesn't duplicate.
"""

import secrets as pysecrets

from app.aws.clients import get_client
from app.config import get_settings

PARAMETERS: dict[str, str] = {
    "s3/bucket-name": "inbox-uploads",
    "llm/model-name": "llama3.2:3b",
    "features/email-digest": "false",
}


def seed_parameters(env: str) -> None:
    ssm = get_client("ssm")
    for suffix, value in PARAMETERS.items():
        name = f"/docinbox/{env}/{suffix}"
        # Overwrite=True makes this idempotent: re-seeding refreshes
        # values instead of failing with ParameterAlreadyExists.
        ssm.put_parameter(Name=name, Value=value, Type="String", Overwrite=True)
        print(f"  param  {name} = {value}")


def seed_signing_key(env: str) -> None:
    sm = get_client("secretsmanager")
    name = f"docinbox/{env}/signing-key"
    try:
        sm.create_secret(Name=name, SecretString=pysecrets.token_urlsafe(32))
        print(f"  secret {name} (created)")
    except sm.exceptions.ResourceExistsException:
        # Unlike parameters, we do NOT overwrite an existing secret:
        # a signing key that changes on every seed would invalidate
        # everything previously signed with it.
        print(f"  secret {name} (already exists, kept)")


def main() -> None:
    env = get_settings().app_env
    print(f"Seeding docinbox resources for env '{env}'...")
    seed_parameters(env)
    seed_signing_key(env)
    print("Done.")


if __name__ == "__main__":
    main()
```

(Remember to add an empty `bootstrap/__init__.py`, so that `python -m bootstrap.seed` resolves.)

The two resource types are seeded with different idempotency strategies, and this is intentional. The parameters are overwritten (`Overwrite=True`), because the seed script is the source of truth for configuration values, and re-seeding should converge them to what the script says. The secret, on the other hand, is created only if it does not exist yet: if the signing key changed on every seed run, everything signed with the previous key would become invalid.

We add the corresponding target to the `Makefile`:

```makefile
seed:          ## Seed SSM parameters + secrets into MiniStack
	python -m bootstrap.seed
```

Now we are ready to run it (with MiniStack up and the virtual environment active):

```bash
make seed
# Seeding docinbox resources for env 'dev'...
#   param  /docinbox/dev/s3/bucket-name = inbox-uploads
#   param  /docinbox/dev/llm/model-name = llama3.2:3b
#   param  /docinbox/dev/features/email-digest = false
#   secret docinbox/dev/signing-key (created)
# Done.
```

After the above command is executed successfully, we can verify the results from the AWS CLI. As in Part 1, the seed script and the AWS CLI are looking at the same emulated AWS:

```bash
aws --endpoint-url http://localhost:4566 ssm get-parameters-by-path \
    --path /docinbox/dev --recursive --query "Parameters[].Name"
# [
#     "/docinbox/dev/features/email-digest",
#     "/docinbox/dev/llm/model-name",
#     "/docinbox/dev/s3/bucket-name"
# ]

aws --endpoint-url http://localhost:4566 secretsmanager list-secrets \
    --query "SecretList[].Name"
# [
#     "docinbox/dev/signing-key"
# ]
```

If we run `make seed` a second time, nothing breaks, and the secret reports "kept".

### 3. The typed app config and its loader

Next, we build the application-side counterpart. One module owns fetching the configuration tree, validating it, and holding the secret:

`app/app_config.py`:

```python
"""Application config, loaded from SSM Parameter Store + Secrets Manager.

Bootstrap settings (endpoint, region, env) come from the environment —
see config.py. Everything here comes from AWS at runtime.
"""

import time

from pydantic import BaseModel, SecretStr

from app.aws.clients import get_client
from app.config import get_settings


class AppConfig(BaseModel):
    """Validated application configuration."""

    bucket_name: str
    llm_model: str
    email_digest_enabled: bool
    signing_key: SecretStr  # renders as '**********' in logs and repr


def _load_parameters(prefix: str) -> dict[str, str]:
    """Fetch every parameter under `prefix` as {relative-name: value}."""
    ssm = get_client("ssm")
    params: dict[str, str] = {}
    paginator = ssm.get_paginator("get_parameters_by_path")
    for page in paginator.paginate(Path=prefix, Recursive=True, WithDecryption=True):
        for param in page["Parameters"]:
            params[param["Name"].removeprefix(prefix + "/")] = param["Value"]
    return params


def _load_signing_key(env: str) -> SecretStr:
    sm = get_client("secretsmanager")
    resp = sm.get_secret_value(SecretId=f"docinbox/{env}/signing-key")
    return SecretStr(resp["SecretString"])


def load_app_config() -> AppConfig:
    """One startup-time round trip to SSM + one to Secrets Manager."""
    env = get_settings().app_env
    params = _load_parameters(f"/docinbox/{env}")
    return AppConfig(
        bucket_name=params["s3/bucket-name"],
        llm_model=params["llm/model-name"],
        email_digest_enabled=params.get("features/email-digest") == "true",
        signing_key=_load_signing_key(env),
    )


class ConfigCache:
    """Holds AppConfig in memory, re-fetching when older than the TTL."""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._config: AppConfig | None = None
        self._loaded_at = 0.0

    def get(self) -> AppConfig:
        if self._config is None or time.monotonic() - self._loaded_at > self._ttl:
            self._config = load_app_config()
            self._loaded_at = time.monotonic()
        return self._config

    def invalidate(self) -> None:
        """Force a re-fetch on the next get() — used by tests (and an exercise)."""
        self._config = None
```

A few implementation details are worth pointing out:

- `get_parameters_by_path` returns at most 10 parameters per page. The paginator handles this for us, so the loader does not silently break once we add an eleventh parameter.
- `SecretStr` masks its value in `repr()`, `str()`, and serialized output; reading the real value requires an explicit `.get_secret_value()` call. We will verify this in step 7.
- `AppConfig` is also the place where SSM's string values become typed Python. SSM has no boolean type, so the feature flag is compared against the literal string `"true"`; this loader is the one place where that check is allowed to live.
- The cache uses `time.monotonic()` rather than `time.time()`, so that the TTL arithmetic is not affected when the wall clock jumps (for example, an NTP sync, or a laptop waking from sleep).

### 4. Loading the config in the lifespan

Loading the configuration in the **lifespan** means that a misconfigured application fails at startup, before it takes any traffic, instead of failing on the first request that needs the missing value.

`app/main.py` (additions shown; the Part 1 routes stay):

```python
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from botocore.exceptions import BotoCoreError, ClientError

from app.app_config import AppConfig, ConfigCache
from app.aws.clients import get_client
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast: if SSM/Secrets Manager is unreachable or a parameter is
    # missing, the app fails at startup instead of at request time.
    cache = ConfigCache(ttl_seconds=get_settings().config_ttl_seconds)
    cache.get()  # first fetch happens here, at startup
    app.state.config_cache = cache
    yield


app = FastAPI(title="Smart Document Inbox", lifespan=lifespan)


def get_app_config() -> AppConfig:
    """FastAPI dependency: current app config (cached, TTL-refreshed)."""
    return app.state.config_cache.get()
```

Any route can now declare `config: AppConfig = Depends(get_app_config)` and receive validated, cached configuration. In Part 3, the upload endpoint will obtain its bucket name in exactly this way.

### 5. `/whoami` — a first look at STS

In Part 1, we used the STS `GetCallerIdentity` call inside `/healthz` as a reachability probe. Here we give it its own route and look at the actual answer, since Part 8 builds on identity:

```python
@app.get("/whoami")
def whoami() -> dict[str, str]:
    """Who does AWS think we are? (STS GetCallerIdentity)"""
    identity = get_client("sts").get_caller_identity()
    return {
        "account": identity["Account"],
        "arn": identity["Arn"],
        "user_id": identity["UserId"],
    }
```

```bash
curl -s localhost:8000/whoami | jq .
# {
#   "account": "000000000000",
#   "arn": "arn:aws:iam::000000000000:root",
#   "user_id": "AKIAIOSFODNN7EXAMPLE"
# }
```

`GetCallerIdentity` requires no permissions; any valid caller may ask who it is. This makes it useful for health checks and for debugging credential confusion ("which role is this container actually running as?"). MiniStack answers with a placeholder root identity in a fake account. In Part 8, we will assume a scoped worker role, and the answer from this endpoint will change accordingly.

### 6. `/config` — a guarded debug window

Once the configuration comes from a remote service through a cache, `cat .env` no longer answers the question "what configuration is the application actually running with?". A debug route does:

```python
@app.get("/config")
def show_config(config: AppConfig = Depends(get_app_config)) -> dict:
    """Resolved app config — non-secret values only. Guarded by DEBUG_ROUTES."""
    if not get_settings().debug_routes:
        raise HTTPException(status_code=404)  # 404, not 403: don't advertise it
    return {
        "app_env": get_settings().app_env,
        "bucket_name": config.bucket_name,
        "llm_model": config.llm_model,
        "email_digest_enabled": config.email_digest_enabled,
        # SecretStr masks itself — this serializes as "**********".
        "signing_key": str(config.signing_key),
    }
```

```bash
curl -s localhost:8000/config | jq .
# {
#   "app_env": "dev",
#   "bucket_name": "inbox-uploads",
#   "llm_model": "llama3.2:3b",
#   "email_digest_enabled": false,
#   "signing_key": "**********"
# }
```

Two details are worth noting here. First, the guard returns a 404 rather than a 403, so that someone probing for paths does not learn that the route exists. Second, the signing key is serialized as `**********` because of `SecretStr`'s default behavior, not because we remembered to redact it in this route.

### 7. Verifying that the secret never logs

We should not take the masking on faith. To confirm that the secret does not leak, we can open a Python REPL and try to leak it:

```python
>>> from app.app_config import load_app_config
>>> config = load_app_config()
>>> config.signing_key
SecretStr('**********')
>>> print(f"debug: {config}")           # the classic accidental leak
debug: bucket_name='inbox-uploads' llm_model='llama3.2:3b' email_digest_enabled=False signing_key=SecretStr('**********')
>>> config.model_dump_json()            # serialization leak? also no
'{"bucket_name":"inbox-uploads","llm_model":"llama3.2:3b","email_digest_enabled":false,"signing_key":"**********"}'
>>> config.signing_key.get_secret_value()   # the only way in
'Kx3...actual-key...'
```

From the above output, we can see that f-strings, `repr`, and JSON serialization are all closed off by default. `.get_secret_value()` is the only path to the raw value, which also means that `grep -rn "get_secret_value"` finds every place where the codebase touches it.

### 8. The full loop

```bash
make up      # MiniStack
make seed    # parameters + secret (idempotent, run anytime)
make run     # app — lifespan loads config from SSM/Secrets Manager at startup

curl -s localhost:8000/healthz   # {"status":"ok","account":"000000000000"}
curl -s localhost:8000/whoami    # who AWS thinks we are
curl -s localhost:8000/config    # resolved config, secret masked
```

Now we can test the TTL behavior. We change a parameter while the application is running:

```bash
aws --endpoint-url http://localhost:4566 ssm put-parameter \
    --name /docinbox/dev/features/email-digest --value true --overwrite

curl -s localhost:8000/config | jq .email_digest_enabled
# false   ← still cached

# ...within 5 minutes (config_ttl_seconds)...
curl -s localhost:8000/config | jq .email_digest_enabled
# true    ← cache expired, re-fetched, flag flipped — no restart
```

From the above output, we can see that the behavior of a running service changed because we wrote to the configuration service, with the staleness bounded by the TTL. No restart and no redeploy were needed.

## The general pattern

> **Portable takeaway (independent of this particular product):** split the configuration into **bootstrap settings** (where is my config service — from the environment, kept tiny) and **app config** (everything else — fetched from that service at startup into a typed, validated object, cached in memory, and refreshed on a TTL). Name the parameters as hierarchical paths with the environment as a segment, so that one call fetches the whole tree and environments differ by a single prefix. Hold secrets in a wrapper type (`SecretStr`) that masks by default, so that revealing a secret takes explicit code. Finally, make the resource seeding idempotent, so that it can run in every workflow.
{: .prompt-tip }

## In production

> - **The same code path runs.** We unset `AWS_ENDPOINT_URL`, and this exact lifespan loads from the real SSM and Secrets Manager. What we add is IAM permissions: `ssm:GetParametersByPath` on our path prefix, and `secretsmanager:GetSecretValue` on the secret's ARN. Least privilege by path prefix is a large part of why the hierarchical names are worth it.
> - **Rotation is the main reason Secrets Manager exists.** In production, we would attach a rotation Lambda (or use the managed rotation for RDS and similar services). The TTL cache means the application picks up a rotated value within one TTL, so nothing in this part needs to change to survive rotation.
> - **KMS is real there.** `SecureString` and Secrets Manager encryption are backed by real KMS keys with real key policies. Locally, the encryption is emulated (see the fidelity notes below); MiniStack's state directory should not be treated as encrypted at rest.
> - **On Lambda, use the caching extension.** AWS ships the AWS Parameters and Secrets Lambda Extension, a sidecar cache with the same TTL semantics that we built by hand.
> - **Fail-fast startup is a deployment feature.** Because the lifespan raises on missing or invalid configuration, a bad deploy fails its health checks immediately, and the orchestrator keeps the old version serving.
{: .prompt-info }

## MiniStack notes and fidelity

- **All three services share port 4566**, like everything in MiniStack. No new containers and no new configuration are needed.
- **SecureString is emulated, not encrypted.** MiniStack accepts `Type=SecureString` and honors `WithDecryption`, and KMS itself is emulated — but this is not at-rest encryption of your local state. Real KMS key policies, grants, and `kms:Decrypt` permission checks do not apply locally. If a production design hinges on *who can decrypt*, that needs to be verified against the real AWS (Part 10 sets up parity checks for exactly this kind of gap).
- **Rotation mechanics are emulated.** We can call `rotate_secret` and exercise the version stages (`AWSCURRENT`/`AWSPREVIOUS`), but the full production sequence — Secrets Manager invoking a rotation Lambda through its four-step protocol on a schedule — is not something to certify locally. Test the rotation handler logic here; test the wiring against the real AWS.
- **No IAM enforcement.** Any caller can read any parameter or secret locally. A least-privilege policy design cannot be validated here; Part 8 covers how to verify it for real.
- **Quotas and tiers are not enforced.** The real SSM distinguishes standard and advanced tiers (4 KB vs 8 KB values, parameter counts, throughput limits). None of these are hit locally, which also means we will not notice that we are about to hit them in production. The 4 KB standard-tier value limit is the one that bites real projects.
- **`GetCallerIdentity` returns placeholders** (account `000000000000` and a root ARN). This is expected; it becomes interesting in Part 8, when assumed roles change the answer.

## Troubleshooting

- **`ParameterNotFound` at startup.** *Likely cause:* the seed script never ran, or `APP_ENV` does not match (seeded `dev`, but the app reads `prod`). *Fix:* run `make seed`, and confirm that `APP_ENV` in `.env` matches what was seeded.
- **`ResourceNotFoundException` (Secrets Manager).** *Likely cause:* the same as above, for the secret. *Fix:* the same; check with `aws ... secretsmanager list-secrets`.
- **The app starts, but `/config` shows stale values.** *Likely cause:* the TTL cache is still fresh; this is by design. *Fix:* wait out `config_ttl_seconds`, restart, or build the reload endpoint (exercise 3).
- **`/config` returns 404.** *Likely cause:* `DEBUG_ROUTES` is not set to `true`. *Fix:* set it in `.env`.
- **`get_parameters_by_path` returns an empty result.** *Likely cause:* a path typo; SSM prefix matching is exact on segments. *Fix:* compare `aws ssm get-parameters-by-path --path /docinbox/dev --recursive` against the prefix in the code.
- **Seed script fails with `ModuleNotFoundError: app`.** *Likely cause:* it was run as `python bootstrap/seed.py` from the wrong place. *Fix:* run it as a module from the repository root: `python -m bootstrap.seed`.

## Weekend exercises

1. **Move a hardcoded value into SSM.** The `title="Smart Document Inbox"` in `main.py` is configuration. Seed `/docinbox/dev/app/title`, add it to `AppConfig`, and use it. Note the shape of the change: the seed script, a model field, and one call site — the loader itself does not change.
2. **Add a secret and prove it never logs.** Seed a fake third-party API key (`docinbox/dev/external-api-key`) in Secrets Manager, add it to `AppConfig` as a `SecretStr`, and then try to leak it: f-string it, `model_dump_json()` it, and log the whole config object at startup. Every path should show `**********`.
3. **Build the reload endpoint.** Add `POST /config/reload` (guarded like `/config`) that calls `cache.invalidate()`, so that a configuration change propagates on demand instead of waiting out the TTL. Flip the `email-digest` flag via the CLI and watch it change immediately.
4. **Explore SecureString.** Re-seed `llm/model-name` as `Type=SecureString`. Fetch it with and without `WithDecryption=True` from the CLI and compare the outputs. Then check whether the app noticed the change. (It should not have — the loader already passes `WithDecryption=True`.)
5. **Break it on purpose.** Delete a required parameter (`aws ssm delete-parameter --name /docinbox/dev/s3/bucket-name`) and start the app. Read the full startup failure, so that you recognize it later — this is the fail-fast behavior from step 4 doing its job. Then run `make seed` to fix it.

## What is next

In this part, we have demonstrated how to load application configuration from SSM Parameter Store and a secret from Secrets Manager at startup, cache them with a TTL, and keep the secret out of the logs with `SecretStr`. The application now configures itself from AWS the way a production service does, and it knows that its bucket is called `inbox-uploads`. It is time to actually put documents into it.

**Part 3 — Object Storage: S3 in a FastAPI app.** We will build real upload and download endpoints, streaming responses, presigned URLs (so that clients transfer bytes without them flowing through our API), versioning, and lifecycle rules — and we will take the first step into async with `aioboto3`, now that there is real I/O worth overlapping. The bucket name will come from the configuration layer we just built.

The source code used in this part is available in the companion GitHub repository: [github.com/chuan2019/docinbox](https://github.com/chuan2019/docinbox) (tag `part-02`).

*See you next weekend.*
