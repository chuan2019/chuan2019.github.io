---
title: "Building a Smart Document Inbox on Local AWS — Part 3: Object Storage"
date: 2026-07-26 10:00:00 +0000
categories: [AWS, MiniStack]
tags: [aws, ministack, s3, fastapi, boto3, aioboto3, python, presigned-urls, streaming, async]
description: "Real upload and download endpoints on S3 — streaming responses, presigned URLs, versioning, and lifecycle rules — plus the series' first step into async with aioboto3, all locally with MiniStack."
---

*S3 in a FastAPI app: uploads, downloads, presigned URLs, and going async*

> **About this series.** Across this series, we are going to build one real and useful product — a **Smart Document Inbox**. The idea is simple: a user drops in a PDF or an article, and it is stored, understood by a local LLM (a summary, together with its topics and entities), made browsable, and finally emailed back to the user as an AI-generated summary. We build the whole thing on a laptop, with **no AWS account and no cloud spend**, using [MiniStack](https://ministack.org/) to emulate AWS and [Ollama](https://ollama.com) to run the LLM locally.
>
> **Catching up?** In [Part 2](/posts/smart-document-inbox-part-2-configuration-and-secrets/), we moved `docinbox`'s configuration into SSM Parameter Store and its first secret into Secrets Manager, and loaded both at startup into a typed `AppConfig` object. This part assumes that skeleton is in place, and it uses `config.bucket_name` directly.
{: .prompt-info }

## Where Part 2 left us

By the end of Part 2 the app knows its bucket is called `inbox-uploads`, because it reads that from SSM at startup. And that's about it. The bucket itself was created by hand with a curl command back in Part 1's exercises, and no code has stored a document in it yet.

This part is where `docinbox` starts doing the thing in its name: take a document in, put it in S3, hand it back out. Along the way we pick up the S3 features you'd want in a real upload/download path (presigned URLs, streaming, versioning, lifecycle rules), and we write the series' first async code, because moving file bytes is the first place in this app where async is worth the trouble.

## What we will cover in this part

- Simple (`PutObject`) vs multipart upload, and why we let `boto3`'s transfer manager choose
- Presigned URLs: a time-limited, signed URL that lets a client fetch from S3 directly instead of going through our API
- Streaming responses, so a large document never has to sit fully in memory
- Versioning and lifecycle rules at the bucket level
- Path-style vs virtual-hosted-style addressing, and why MiniStack needs the former
- Converting the upload/download path to `aioboto3`, and when that's worth doing

## Prerequisites

- The Part 2 skeleton, running (`make up`, `make seed`, `make run`, and `/config` returns the resolved config)
- Two new Python packages: `aioboto3` (an async `boto3`) and `python-multipart` (FastAPI needs it to parse file uploads)
- No new MiniStack services. S3 has been listening on port 4566 since Part 1.

---

## Concepts

### Simple upload vs multipart upload

S3 gives you two ways to upload. `PutObject` sends the whole object in one HTTP request, which is fine when you already have all the bytes in memory anyway. Multipart upload splits the object into parts (`CreateMultipartUpload`, then a series of `UploadPart` calls, then `CompleteMultipartUpload`). You're forced into multipart above 5 GB, but it's useful well below that: parts can upload in parallel, and if one part fails you retry just that part instead of re-sending the whole file.

The good news is that we don't have to write any of this. `upload_fileobj` looks at the object's size and switches to multipart above a threshold (8 MB by default). It's the same call whether the document is a 40 KB text file or a 400 MB PDF, and I see no reason to reimplement something `s3transfer` already gets right.

### Presigned URLs

A presigned URL is an ordinary S3 URL with a signature and an expiry baked into the query string. Generating one (`generate_presigned_url`) doesn't touch the network at all; it's an HMAC computed locally with credentials we already have. Whoever holds the URL can do exactly one operation on one key until it expires.

Why bother? Without presigned URLs, every byte of every document passes through our FastAPI process twice, once on the way in and once on the way out. With them, the client talks to S3 directly and our API only ever ships small JSON responses. For a document-heavy app that's a big difference in what the API server has to be sized for.

### Streaming responses

The obvious way to serve a download is to call `get_object`, read the whole body into memory, and return it as `bytes`. That's fine for a 5 KB metadata file. For a 200-page contract it means the full document sits in RAM for the length of the request, once per concurrent download. `StreamingResponse` takes an iterator instead of a value, so bytes flow from S3 through our process to the client in chunks, and the complete document never exists in our memory at any one moment.

### Versioning and lifecycle rules

Turn versioning on for a bucket and overwrites stop being destructive. S3 keeps the old body as a noncurrent version and makes the new one current; `GetObject` without a `VersionId` still returns the current version, so none of the download code has to change. It's a safety net against accidental overwrites and deletes that just sits underneath everything.

One thing that surprised me when I first ran into it: you can't fully turn versioning off again. You can *suspend* a versioned bucket, but it never goes back to "never versioned."

Lifecycle rules are the other half. They're bucket-level policies that act on objects automatically: expire them after N days, transition them to cheaper storage classes, and so on. The one we set up below expires noncurrent versions after 30 days, because otherwise versioning quietly becomes an unbounded storage bill.

### Path-style vs virtual-hosted-style addressing

Real S3 supports two URL shapes for the same object: **virtual-hosted-style** (`https://inbox-uploads.s3.amazonaws.com/<key>`, bucket name as a DNS subdomain) and **path-style** (`https://s3.amazonaws.com/inbox-uploads/<key>`, bucket name as a path segment). AWS has been steering everyone toward virtual-hosted-style for years, but it depends on DNS resolving `inbox-uploads.s3.amazonaws.com`. Nothing on your laptop is going to resolve `inbox-uploads.localhost:4566`, so for MiniStack we tell `boto3` to use path-style:

```python
from botocore.config import Config

Config(s3={"addressing_style": "path"})
```

This matters mostly for presigned URLs. Get the addressing style wrong and the signed URL points at a host that doesn't exist, so the request hangs or fails to connect before MiniStack ever sees it. There's no helpful signature error, because the signature never gets checked. I lost a good half hour to this the first time; there's an entry in Troubleshooting for it.

---

## Build

The target layout after this part (new/changed files marked):

```
docinbox/
├── app/
│   ├── main.py               # + async S3 client in lifespan, /documents endpoints
│   ├── config.py             # unchanged
│   ├── app_config.py         # unchanged
│   └── aws/
│       ├── clients.py        # unchanged — still used by /healthz, /whoami, bootstrap
│       └── aclients.py       # NEW: the async client factory (aioboto3)
├── bootstrap/
│   └── seed.py                # + seed_bucket: create bucket, enable versioning, lifecycle rule
├── docker-compose.yml         # unchanged — S3_PERSIST=1 was already set in Part 1
├── requirements.txt           # + aioboto3, python-multipart
└── ...
```

### 1. New dependencies

`requirements.txt` gains two lines:

```
aioboto3>=13.0
python-multipart>=0.0.9
```

```bash
pip install -r requirements.txt
```

Note that `python-multipart` is not optional. FastAPI raises at import time if a route declares `UploadFile` and the package is missing.

### 2. The async client factory

Part 1's `get_client()` stays exactly as it is. `/healthz`, `/whoami`, and `bootstrap/seed.py` keep using it, since none of those do enough I/O for async to matter. What we add is an async counterpart, used only where we're about to move real document bytes.

`aioboto3` clients differ from `boto3` clients in one important way: they hold an open `aiohttp` session, so they have to be used as an **async context manager** (`async with session.client(...) as s3:`) rather than constructed once and cached as a bare object the way `get_client` caches a `boto3.client`. The natural place to open one for the lifetime of the app is the FastAPI **lifespan**, with an `AsyncExitStack` to make sure it closes on shutdown.

`app/aws/aclients.py`:

```python
"""The async counterpart to clients.py — for the I/O worth overlapping.

boto3 clients (clients.py) are cheap to cache and reuse everywhere.
aioboto3 clients hold an open aiohttp session and must be opened as an
async context manager, so this module hands one out tied to an
AsyncExitStack the caller owns (in practice: the app lifespan).
"""

from contextlib import AsyncExitStack
from typing import Any

import aioboto3
from botocore.config import Config

from app.config import get_settings

# Path-style addressing: with the virtual-hosted default, a presigned URL
# would point at http://inbox-uploads.localhost:4566/..., which nothing
# resolves. Path-style keeps the bucket name in the path instead.
_S3_ADDRESSING = Config(s3={"addressing_style": "path"})

_session = aioboto3.Session()


def _client_kwargs() -> dict[str, Any]:
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "region_name": settings.aws_region,
        "aws_access_key_id": settings.aws_access_key_id,
        "aws_secret_access_key": settings.aws_secret_access_key,
        "config": _S3_ADDRESSING,
    }
    if settings.aws_endpoint_url:
        kwargs["endpoint_url"] = settings.aws_endpoint_url
    return kwargs


async def open_async_client(stack: AsyncExitStack, service: str) -> Any:
    """Open an async client for `service`, tied to `stack`'s lifetime.

    Call this once (the app lifespan does) and reuse the client it
    returns — aioboto3 clients are not meant to be opened per request.
    """
    return await stack.enter_async_context(_session.client(service, **_client_kwargs()))
```

### 3. Wiring it into the lifespan

`app/main.py` (the lifespan grows; everything from Part 2 stays):

```python
from contextlib import AsyncExitStack, asynccontextmanager

from app.aws.aclients import open_async_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    cache = ConfigCache(ttl_seconds=get_settings().config_ttl_seconds)
    cache.get()  # fail-fast config load, from Part 2
    app.state.config_cache = cache

    async with AsyncExitStack() as stack:
        app.state.s3 = await open_async_client(stack, "s3")
        yield
    # the stack closes app.state.s3's aiohttp session on shutdown
```

`app.state.s3` is now a live async S3 client for the whole life of the process: one `aiohttp` session shared across every request, instead of one opened and closed per call.

### 4. Uploading a document

```python
import uuid

from fastapi import Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse


@app.post("/documents", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    config: AppConfig = Depends(get_app_config),
) -> dict[str, str]:
    """Upload a document to S3. The document id is a new key, not the filename."""
    document_id = str(uuid.uuid4())
    content_type = file.content_type or "application/octet-stream"

    # upload_fileobj picks simple vs multipart based on size — we don't.
    await app.state.s3.upload_fileobj(
        file.file,
        config.bucket_name,
        document_id,
        ExtraArgs={
            "ContentType": content_type,
            "Metadata": {"filename": file.filename or "unnamed"},
        },
    )
    return {"document_id": document_id, "filename": file.filename, "content_type": content_type}
```

The key is a fresh UUID rather than the original filename. That way two people uploading `invoice.pdf` don't collide, though it does mean we need somewhere to remember what a given id *is*. For now that somewhere is the object's own S3 metadata. Part 4 replaces this with a proper DynamoDB record.

`file.file` is FastAPI's `SpooledTemporaryFile` (small uploads stay in memory, large ones spill to disk on their own), and we hand it straight to `upload_fileobj` rather than reading it into a `bytes` first. Reading it fully up front would work fine for a demo, but it would also throw away most of what this section is trying to accomplish.

### 5. Listing documents

We don't have a database yet, so "list documents" means asking S3 what's in the bucket. This is a placeholder that Part 4 replaces with a real query:

```python
@app.get("/documents")
async def list_documents(config: AppConfig = Depends(get_app_config)) -> dict[str, list[dict]]:
    resp = await app.state.s3.list_objects_v2(Bucket=config.bucket_name)
    documents = [
        {
            "document_id": obj["Key"],
            "size": obj["Size"],
            "last_modified": obj["LastModified"].isoformat(),
        }
        for obj in resp.get("Contents", [])
    ]
    return {"documents": documents}
```

### 6. Downloading a document — streamed

```python
from botocore.exceptions import ClientError


@app.get("/documents/{document_id}/download")
async def download_document(
    document_id: str,
    config: AppConfig = Depends(get_app_config),
) -> StreamingResponse:
    try:
        obj = await app.state.s3.get_object(Bucket=config.bucket_name, Key=document_id)
    except app.state.s3.exceptions.NoSuchKey as exc:
        raise HTTPException(status_code=404, detail="document not found") from exc

    filename = obj.get("Metadata", {}).get("filename", document_id)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    if "ContentLength" in obj:
        headers["Content-Length"] = str(obj["ContentLength"])

    return StreamingResponse(
        obj["Body"].iter_chunks(),
        media_type=obj.get("ContentType", "application/octet-stream"),
        headers=headers,
    )
```

`obj["Body"]` from an `aioboto3` response is an async streaming body, and `iter_chunks()` gives you an async generator of byte chunks, which happens to be exactly the shape `StreamingResponse` wants. At no point does the whole document exist as a single `bytes` object in our process.

### 7. Downloading a document — presigned URL

The streamed route works, but every byte still passes through our API process. The alternative is to hand the client a presigned URL and let it fetch from S3 directly.

```python
@app.get("/documents/{document_id}/download-url")
async def download_document_url(
    document_id: str,
    config: AppConfig = Depends(get_app_config),
) -> dict[str, str | int]:
    # HeadObject confirms the key exists before we hand out a URL for it.
    try:
        await app.state.s3.head_object(Bucket=config.bucket_name, Key=document_id)
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "404":
            raise HTTPException(status_code=404, detail="document not found") from exc
        raise

    url = await app.state.s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": config.bucket_name, "Key": document_id},
        ExpiresIn=300,
    )
    return {"url": url, "expires_in": 300}
```

Two small things worth noting. First, `generate_presigned_url` gets `await`-ed here even though signing does no network I/O; `aioboto3` wraps every client method as a coroutine for consistency, and this one just returns immediately. Second, the 300-second expiry is short on purpose. A presigned URL works for anyone who has it, so the window should be just long enough for the client to start the download.

### 8. Seeding the bucket: versioning + lifecycle

Until now the bucket has only ever been created by hand. Time to fold it into the same idempotent `bootstrap/seed.py` from Part 2, alongside the parameters and the secret. We reuse the bucket name already defined in `PARAMETERS`, so the config and the seed script can't drift apart:

```python
def seed_bucket(env: str) -> None:
    s3 = get_client("s3")
    bucket = PARAMETERS["s3/bucket-name"]

    try:
        s3.create_bucket(Bucket=bucket)
        print(f"  bucket  {bucket} (created)")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"  bucket  {bucket} (already exists, kept)")

    s3.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})

    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "expire-noncurrent-versions",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},
                    "NoncurrentVersionExpiration": {"NoncurrentDays": 30},
                }
            ]
        },
    )
    print(f"  bucket  {bucket}: versioning enabled, lifecycle rule applied")


def main() -> None:
    env = get_settings().app_env
    print(f"Seeding docinbox resources for env '{env}'...")
    seed_parameters(env)
    seed_signing_key(env)
    seed_bucket(env)
    print("Done.")
```

The bucket gets the same create-only-if-missing treatment as the secret in Part 2, so re-running the seed script won't fail just because the bucket already exists. `put_bucket_versioning` and `put_bucket_lifecycle_configuration` don't need any of that: they're declarative sets, and calling them again simply reasserts the same configuration.

### 9. The full loop

```bash
make up      # MiniStack
make seed    # parameters + secret (Part 2) + bucket, versioning, lifecycle (this part)
make run     # app — opens the async S3 client in the lifespan
```

Upload a document and note its id:

```bash
curl -s -X POST localhost:8000/documents -F "file=@contract.pdf;type=application/pdf" | jq .
# {
#   "document_id": "6f1a9e2e-...-...",
#   "filename": "contract.pdf",
#   "content_type": "application/pdf"
# }
```

List it, download it streamed, and diff the result against the original:

```bash
curl -s localhost:8000/documents | jq .
# {"documents":[{"document_id":"6f1a9e2e-...","size":48213,"last_modified":"2026-07-20T..."}]}

curl -s localhost:8000/documents/6f1a9e2e-.../download -o out.pdf
diff contract.pdf out.pdf && echo "identical"
# identical
```

Now the presigned route. Note that the second `curl` below talks straight to MiniStack:

```bash
curl -s localhost:8000/documents/6f1a9e2e-.../download-url | jq .
# {"url":"http://localhost:4566/inbox-uploads/6f1a9e2e-...?X-Amz-Algorithm=...","expires_in":300}

curl -s "http://localhost:4566/inbox-uploads/6f1a9e2e-...?X-Amz-Algorithm=..." -o out2.pdf
diff contract.pdf out2.pdf && echo "identical, and our API process never touched these bytes"
```

You could kill the FastAPI process entirely at this point and that second `curl` would still work, since it never touches our API. Try it if you don't believe me.

Versioning, on a separate manual key (so we don't need to reuse a real document id to see it):

```bash
echo "v1" | aws --endpoint-url http://localhost:4566 s3 cp - s3://inbox-uploads/demo.txt
echo "v2" | aws --endpoint-url http://localhost:4566 s3 cp - s3://inbox-uploads/demo.txt

aws --endpoint-url http://localhost:4566 s3api list-object-versions \
    --bucket inbox-uploads --prefix demo.txt --query "Versions[].{Id:VersionId,Latest:IsLatest}"
# [
#     {"Id": "null" or a real version id, "Latest": true},
#     {"Id": "...", "Latest": false}
# ]
```

Two versions of the same key now exist, and `GET /documents`, which always reads the current version, is unaffected.

Finally, persistence. `S3_PERSIST=1` has been set since Part 1, but this is the first time we have bytes worth losing:

```bash
docker compose restart ministack
curl -s localhost:8000/documents | jq .
# the same document is still listed — the object survived the restart
```

---

## The general pattern

> **The portable version of this, if you're building something else:** an endpoint that moves bytes to and from an object store shouldn't default to proxying everything through your API. Offer a presigned-URL path for large transfers. Stream rather than buffer on whatever side does go through your process. And reach for an async client at the specific boundary where a blocking call would sit on a thread-pool worker for the whole transfer, rather than rewriting code that was working fine synchronously.
{: .prompt-tip }

## In production

> - We put `put_bucket_versioning` and `put_bucket_lifecycle_configuration` in a seed script because it's convenient locally. In a real deployment that's Terraform or CloudFormation's job. The app shouldn't be mutating its own bucket's config at runtime.
> - The 300-second presigned expiry isn't just a round number I picked; production systems usually go tighter. Whatever you choose, scope the URL to one key and one operation. Never a prefix, never a wildcard.
> - Tune the multipart threshold once you know your real traffic. 8 MB is a fine general default, but on a fast, high-latency link, dropping it via `TransferConfig` lets more parts upload in parallel, and that can be a real throughput win.
> - Async pays off at concurrency. A blocking `boto3` call sitting in FastAPI's thread pool is fine at low volume; the trouble starts when you have enough concurrent uploads and downloads that the default pool size (40, for Uvicorn/Starlette) becomes the bottleneck rather than S3 itself. If you're nowhere near that, honestly, the sync version is fine.
{: .prompt-info }

## MiniStack notes and fidelity

- `S3_PERSIST=1` and the volume mounts from Part 1 are the whole reason the restart demo above works. Object bytes and bucket-level config both survive `docker compose restart`. If you're relying on some more obscure piece of S3 metadata surviving a restart, don't assume; go check.
- Path-style addressing is a hard requirement here, not a preference. Drop the `Config(s3={"addressing_style": "path"})` and presigned URLs get signed for a virtual-hosted host that nothing on your machine resolves. As mentioned above, the failure mode is a connection failure with no useful error, and it's annoying to debug the first time you hit it.
- Lifecycle rules have a timing gap worth knowing about. Real S3 evaluates them in a batch job that runs roughly once a day, so an `Expiration` of one day doesn't mean the object disappears the moment the clock rolls over. Treat the lifecycle exercise below as a configuration exercise; watching it fire needs real AWS and more patience than a weekend gives you.
- No IAM enforcement, still, same as every part so far. Any caller can read or write anything in the bucket. Part 8 is where this gets fixed.

## Troubleshooting

- **`ImportError` mentioning `python-multipart` when you hit `/documents`.** The dependency from step 1 is missing. `pip install -r requirements.txt`.
- **`NoSuchBucket` on upload.** You probably pulled this part's changes but didn't re-run `make seed`. Run it, then sanity-check with `aws --endpoint-url http://localhost:4566 s3 ls`.
- **The presigned URL 404s or connects to the wrong host.** This is the addressing-style problem: either the `Config` from step 2 got dropped, or you copied a URL that was generated before that change went in. Re-request `/documents/{id}/download-url` and look at the host in the returned URL. It should be `localhost:4566/inbox-uploads/...`, not `inbox-uploads.localhost:4566/...`.
- **`RuntimeError: Session is closed` on the second request after a hot-reload.** `uvicorn --reload` restarted the app, and the async S3 client from the old lifespan went with it. This is normal during development; retry once the reload settles, or do a real restart.
- **Downloaded file differs from the upload.** Before assuming data corruption, check whether a client mangled line endings (usually a `Content-Type`/`Content-Disposition` mismatch). Compare with `diff` or `sha256sum` rather than eyeballing the file.

## Weekend exercises

1. **Presign an upload, not just a download.** Add `GET /documents/upload-url` returning a presigned URL for `put_object` with a fresh document id, so a client can upload straight to S3 without the bytes ever reaching our API. Notice what breaks: the `Metadata={"filename": ...}` trick has nowhere to run anymore, since we never see the upload happen.
2. Add server-side encryption. Drop `"ServerSideEncryption": "AES256"` into the `ExtraArgs` in `upload_document`, then confirm it took by checking the `ServerSideEncryption` field on a `head_object` response.
3. Add a real expiration rule on objects under a `tmp/` prefix, in addition to the noncurrent-version one. You won't see it fire this weekend (see the fidelity note above about lifecycle timing), but it's worth writing the rule correctly and noting it in Appendix C to check against real AWS someday.
4. **Force a multipart upload and go watch it happen.** Pass `TransferConfig(multipart_threshold=1024 * 1024)` into `upload_fileobj`, upload something over 1 MB, and set `LOG_LEVEL=DEBUG` on the MiniStack container to watch the individual `UploadPart` calls scroll by.

## What is next

`docinbox` can now take a document in and hand it back out, either streamed through the API or via a presigned URL that skips it entirely, with versioning and a lifecycle rule underneath. What it still can't do is answer anything about *who* a document belongs to or what state it's in. Right now "list documents" just means asking S3 what keys happen to exist, and that falls over the moment you want "show me Alice's contracts."

**Part 4 — The Data Layer: DynamoDB.** We'll design a single-table schema around `docinbox`'s real access patterns, build a typed repository on top of it, add a GSI for documents-by-owner, and leave room in the schema for the LLM-derived fields Part 7 will eventually populate. The `list_objects_v2` placeholder from this part gets replaced with an actual query.

The source code used in this part is available in the companion GitHub repository: [github.com/chuan2019/docinbox](https://github.com/chuan2019/docinbox) (tag `part-03`).

*See you next weekend.*
