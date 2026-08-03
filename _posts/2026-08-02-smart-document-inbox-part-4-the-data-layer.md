---
title: "Building a Smart Document Inbox on Local AWS — Part 4: The Data Layer"
date: 2026-08-02 10:00:00 +0000
categories: [AWS, MiniStack]
tags: [aws, ministack, dynamodb, single-table-design, gsi, ttl, transactions, dynamodb-streams, fastapi, pydantic, python]
description: "A real data layer for docinbox: DynamoDB single-table design around actual access patterns, a typed repository, a GSI for documents-by-owner, TTL, transactions, and a first look at Streams — all locally with MiniStack."
---

*DynamoDB: single-table design, a typed repository, GSIs, TTL, transactions, and a first look at Streams*

> **About this series.** Across this series, we are going to build one real and useful product, a **Smart Document Inbox**. The idea is simple: a user drops in a PDF or an article, and it is stored, understood by a local LLM (a summary, together with its topics and entities), made browsable, and finally emailed back to the user as an AI-generated summary. We build the whole thing on a laptop, with **no AWS account and no cloud spend**, using [MiniStack](https://ministack.org/) to emulate AWS and [Ollama](https://ollama.com) to run the LLM locally.
>
> **Catching up?** In [Part 3](/posts/smart-document-inbox-part-3-object-storage/), we built the upload and download paths on S3 (streamed responses, presigned URLs, versioning, a lifecycle rule) and opened the series' first async client (`aioboto3`) in the FastAPI lifespan. This part assumes that skeleton is in place, and it extends the same lifespan and the same seed script.
{: .prompt-info }

---

## Where Part 3 left us

`docinbox` can take a document in and hand it back out, but it cannot answer any question *about* its documents: who uploaded this one, whether it is still being processed, or which documents Alice uploaded last week. Right now the "list documents" endpoint is just a `list_objects_v2` call, which asks S3 which keys exist and returns them. The only metadata we keep, the original filename, is stored inside the S3 object's own metadata, where nothing can query it.

In this part, we give `docinbox` a real data layer. We design a DynamoDB table around the application's actual access patterns, put a typed repository in front of it, and replace the Part 3 placeholder with a real query. Worth noting: we also add the LLM-derived fields (summary, topics, entities, language, and document type) to the schema now, even though they will stay empty until the Part 7 Lambda starts writing them. It costs almost nothing at this point, and it saves a schema change later.

## What we will cover in this part

- Partition keys, sort keys, and item collections: the DynamoDB data model in one page
- **Single-table design**, and the rule that makes it manageable: *design for access patterns, not entities*
- A **GSI** for "list documents by owner", and why ISO-8601 timestamps make good sort keys
- **TTL** for transient records that should clean themselves up
- **Transactions** (`TransactWriteItems`) for a status update that must not race
- A typed `Repository` class, so route handlers never see a partition key
- A side-quest into **DynamoDB Streams**: what they are, and why our pipeline will use SNS→SQS instead

## Prerequisites

- The Part 3 skeleton, running (`make up`, `make seed`, `make run`, and an upload round-trips through S3)
- Nothing new to install, and no new MiniStack services. DynamoDB (and DynamoDB Streams) has been listening on port 4566 since Part 1; both show up in the `/healthz` service listing

---

## Concepts

### Keys, items, and item collections

A DynamoDB table has no schema beyond its key. Every item must carry the **partition key** (which determines where the item lives), and, if the table defines one, the **sort key** (which orders items *within* a partition). Everything else is optional attributes that individual items may or may not have.

All the items sharing one partition key form an **item collection**, sorted by the sort key, and a `Query` reads from exactly one collection: efficiently, in order, and with conditions on the sort key (`begins_with`, ranges). In practice, DynamoDB modeling comes down to one question: can every important read be phrased as "give me a slice of one item collection"?

What we give up: joins, and any real ad-hoc querying (`Scan` reads the entire table, so it does not count). There is no way to add an index later and let an optimizer figure things out. We get exactly the queries we designed keys for.

### Design for access patterns, not entities

Relational modeling starts with entities: one table per noun, normalized, and queries come later. DynamoDB modeling runs the other way. We start by writing down every question the application needs to ask, and then design keys so that each question is one `GetItem` or one `Query`. If a question has no key shape, the application cannot ask it.

For `docinbox`, the list is short:

| # | Access pattern | Where it's needed |
|---|---|---|
| 1 | Get one document's metadata by id | `GET /documents/{id}`, and every worker touch |
| 2 | List an owner's documents, newest first | `GET /documents` — the real version of Part 3's placeholder |
| 3 | Read a document's status history | debugging the pipeline in Parts 5–7 |
| 4 | Update a document's status without races | the worker (Part 5) and the Lambda (Part 7) |
| 5 | Filter an owner's documents by topic/type | after Part 7, when the LLM fields exist |

Patterns 1–4 are implemented in this part. Pattern 5 is the reason the schema gets the LLM fields now; the queries for it come in Part 7, when there is something in them.

### Single-table design

We will put documents *and* their status-history events into **one table**, distinguished by key prefixes (`DOC#...`, `EVENT#...`) rather than by separate tables. This is called single-table design, and it can look strange at first if you come from a relational background. It is, however, the normal practice in DynamoDB, for a few reasons. Related items that share a partition key can be fetched in one query: a document and its whole event history form one item collection, so pattern 3 comes for free. One table also means one set of capacity settings, alarms, and stream wiring instead of several. And since DynamoDB has no joins anyway, spreading the data across multiple tables does not buy us much.

The cost is readability: the table cannot be understood without the key-design document next to it, and generic attribute names like `PK`/`GSI1PK` mean nothing by themselves. This is why the repository layer below exists: key construction lives in one module, and the rest of the app deals with `DocumentRecord` objects that never mention `PK`.

Our key design, in full:

| Item | `PK` | `SK` | `GSI1PK` | `GSI1SK` |
|---|---|---|---|---|
| Document metadata | `DOC#<id>` | `META` | `OWNER#<owner>` | `UPLOADED#<iso-ts>#<id>` |
| Status event | `DOC#<id>` | `EVENT#<iso-ts>` | — | — |

### Global Secondary Indexes

The main table answers "by document id". It cannot answer pattern 2, "documents by owner", because owner is just a regular attribute, and attributes are not queryable. A **Global Secondary Index** solves this: it is a second, automatically maintained copy of the data with a *different* key shape. We give documents `GSI1PK = OWNER#<owner>` and `GSI1SK = UPLOADED#<timestamp>#<id>`, so querying `GSI1` for `OWNER#alice@example.com` returns Alice's documents in upload order.

Two details are worth noting here:

- **ISO-8601 timestamps sort correctly as strings**, as long as they are fixed-width and in one timezone (we use UTC everywhere). That is why `UPLOADED#2026-08-01T09:15:00+00:00#...` works as a sort key; `ScanIndexForward=False` then gives newest-first.
- **GSIs are eventually consistent** in real AWS. A write to the table usually propagates to the index in milliseconds, but there is no guarantee, and a read from the GSI immediately after a write may miss the new item. MiniStack does not reproduce this behavior, so it is easy to forget until production; it comes back in the fidelity notes below.

Status events do not carry the GSI attributes at all. A GSI is **sparse**: items without the index keys simply do not appear in it. That is deliberate; the owner index should contain only documents.

### TTL: records that clean themselves up

The status events from pattern 3 are useful while we are building and debugging the pipeline, but we do not want them to accumulate forever, and a dedicated cleanup job for them is not worth the effort. DynamoDB **TTL** handles this housekeeping: nominate one attribute (`expires_at`, holding a Unix epoch in seconds), and DynamoDB deletes items whose time has passed.

One important limitation: the deletion is **lazy**. Real AWS scans for expired items in the background and typically deletes them within a day or two after expiry, so an expired item can still appear in queries until then. Code that must not see expired items has to filter them out itself; TTL is only a cleanup mechanism, not a visibility rule.

### Transactions and conditional writes

From Part 5 on, documents move through a status state machine: `uploaded → processing → processed` (or `failed`). Two workers grabbing the same document, or one retry racing the original, must not both "win" a status change, and when a change does happen, we want the audit event written together with it, atomically.

DynamoDB provides two tools here, and we use both at once:

- A **condition expression** makes a single write conditional on the item's current state (`status = :expected`); if the condition fails, the write fails, atomically.
- **`TransactWriteItems`** groups up to 100 writes into one all-or-nothing unit: either every write commits, or none does, and any condition failure cancels the whole transaction.

Our status update is one transaction: update the document's status (conditional on the status it is moving from), and put the status event. If the condition fails, both writes are cancelled, so there is no window where the status changed but the event is missing, or vice versa.

### DynamoDB Streams — the side-quest

Enable **Streams** on a table, and every INSERT/MODIFY/REMOVE gets recorded, in order per item, and stays readable for 24 hours. It is AWS's built-in change-data-capture mechanism, and MiniStack emulates it, including Streams→Lambda event-source mappings if you want to explore that route.

Our pipeline will not be built on it, though. A stream record only says that an item changed; the pipeline needs to know that *a document was uploaded and wants processing*, and deriving that meaning from table diffs couples every consumer to the table schema. Streams also fire on every incidental write (a TTL deletion is a REMOVE event too). So starting in Part 5, the application publishes explicit events to SNS at the moments that mean something, and workers consume those. Streams are still the right tool when the data change itself is what you want to react to; replication and cache invalidation are the classic cases.

We will still turn them on and watch them work in this part, because seeing the INSERT/MODIFY/REMOVE records scroll by makes the contrast with Part 5 concrete.

---

## Build

The target layout after this part (new/changed files marked):

```
docinbox/
├── app/
│   ├── main.py               # + dynamodb client in lifespan; /documents endpoints rewritten
│   ├── models.py             # NEW: DocumentRecord, DocumentStatus, status transitions
│   ├── repository.py         # NEW: DocumentRepository — all key logic lives here
│   ├── app_config.py         # + table_name
│   ├── config.py             # unchanged
│   └── aws/
│       ├── clients.py        # unchanged
│       └── aclients.py       # unchanged — it already opens any service
├── bootstrap/
│   ├── seed.py               # + seed_table: table, GSI, Streams, TTL
│   └── tail_stream.py        # NEW: the Streams side-quest
├── docker-compose.yml        # unchanged
└── ...
```

### 1. Seeding the table

The table joins the parameters, the secret, and the bucket in `bootstrap/seed.py`. The table name goes into `PARAMETERS` first, so that the seed script and the application configuration always read the same value (same as the bucket in Part 3):

```python
PARAMETERS: dict[str, str] = {
    "s3/bucket-name": "inbox-uploads",
    "dynamodb/table-name": "docinbox",        # NEW
    "llm/model-name": "llama3.2:3b",
    "features/email-digest": "false",
}
```

Then the seeding function:

```python
from botocore.exceptions import ClientError


def seed_table(env: str) -> None:
    ddb = get_client("dynamodb")
    table = PARAMETERS["dynamodb/table-name"]

    try:
        ddb.create_table(
            TableName=table,
            BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[
                # Only KEY attributes are declared — everything else is schemaless.
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
                {"AttributeName": "GSI1PK", "AttributeType": "S"},
                {"AttributeName": "GSI1SK", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "GSI1",
                    "KeySchema": [
                        {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                        {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
            # The side-quest: record every item change. NEW_AND_OLD_IMAGES
            # captures the full item before and after each write.
            StreamSpecification={
                "StreamEnabled": True,
                "StreamViewType": "NEW_AND_OLD_IMAGES",
            },
        )
        ddb.get_waiter("table_exists").wait(TableName=table)
        print(f"  table  {table} (created)")
    except ddb.exceptions.ResourceInUseException:
        print(f"  table  {table} (already exists, kept)")

    # TTL is configured separately from table creation.
    try:
        ddb.update_time_to_live(
            TableName=table,
            TimeToLiveSpecification={"AttributeName": "expires_at", "Enabled": True},
        )
        print(f"  table  {table}: TTL enabled on 'expires_at'")
    except ClientError as exc:
        # Re-enabling TTL that's already enabled is a ValidationException,
        # which is fine for an idempotent seed — anything else is real.
        if exc.response["Error"]["Code"] != "ValidationException":
            raise
        print(f"  table  {table}: TTL already enabled")


def main() -> None:
    env = get_settings().app_env
    print(f"Seeding docinbox resources for env '{env}'...")
    seed_parameters(env)
    seed_signing_key(env)
    seed_bucket(env)
    seed_table(env)
    print("Done.")
```

Worth noting: `AttributeDefinitions` lists only the four key attributes. `owner`, `status`, `summary`, and everything else the items will carry are never declared anywhere; that is the schemaless side of DynamoDB. The typed side is our job, and it lives in the models below.

Run it:

```bash
make seed
#   ...
#   table  docinbox (created)
#   table  docinbox: TTL enabled on 'expires_at'
```

### 2. The table name joins `AppConfig`

Same drill as the bucket in Parts 2–3: one field, and one line in the loader (`app/app_config.py`):

```python
class AppConfig(BaseModel):
    bucket_name: str
    table_name: str            # NEW
    llm_model: str
    email_digest_enabled: bool
    signing_key: SecretStr


def load_app_config() -> AppConfig:
    env = get_settings().app_env
    params = _load_parameters(f"/docinbox/{env}")
    return AppConfig(
        bucket_name=params["s3/bucket-name"],
        table_name=params["dynamodb/table-name"],     # NEW
        llm_model=params["llm/model-name"],
        email_digest_enabled=params.get("features/email-digest") == "true",
        signing_key=_load_signing_key(env),
    )
```

### 3. Typed models — including room for the LLM

`app/models.py` is new. The five LLM-derived fields are declared now, as optional fields that stay empty until the Part 7 Lambda writes them. When that happens, neither the table, the repository, nor these models will need to change.

```python
"""Domain models for docinbox documents.

The LLM-derived fields (summary, topics, entities, language, doc_type)
are declared now but stay empty until the Part 7 Lambda populates them.
"""

import enum
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class DocumentStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


# The status state machine. update_status() refuses anything not listed here,
# and the DynamoDB condition expression enforces it against races.
VALID_TRANSITIONS: dict[DocumentStatus, set[DocumentStatus]] = {
    DocumentStatus.UPLOADED: {DocumentStatus.PROCESSING},
    DocumentStatus.PROCESSING: {DocumentStatus.PROCESSED, DocumentStatus.FAILED},
    DocumentStatus.FAILED: {DocumentStatus.PROCESSING},   # retry
    DocumentStatus.PROCESSED: set(),                      # terminal
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DocumentRecord(BaseModel):
    """One document's metadata — the app-facing shape, no DynamoDB keys."""

    document_id: str
    owner: str
    filename: str
    content_type: str
    size: int
    status: DocumentStatus = DocumentStatus.UPLOADED
    uploaded_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    # --- LLM-derived fields, written by the Part 7 Lambda ---
    summary: str | None = None
    topics: list[str] = []
    entities: list[str] = []
    language: str | None = None
    doc_type: str | None = None


class StatusEvent(BaseModel):
    """One entry in a document's status history (a TTL'd audit record)."""

    document_id: str
    occurred_at: datetime
    from_status: DocumentStatus
    to_status: DocumentStatus
    detail: str = ""
```

### 4. The repository

`app/repository.py` is the only module that is allowed to know that `PK` exists. Everything above it deals with `DocumentRecord`.

One plumbing decision is worth explaining first. The low-level DynamoDB client speaks a typed wire format (`{"S": "hello"}`, `{"N": "42"}`), while the higher-level `Table` resource accepts plain Python values and hides all of this. The resource API is pleasant to use, until one needs `transact_write_items`, which only exists on the client API. Rather than mixing the two styles, the whole repository uses the client, with the marshalling helpers `boto3` already ships (`TypeSerializer`/`TypeDeserializer`) wrapped in two small functions. The raw format appears once more in the Streams side-quest, where it is useful to recognize.

```python
"""DocumentRepository — every DynamoDB key decision in the app lives here.

Key design (the whole single-table layout, in one place):

    document:      PK=DOC#<id>   SK=META            GSI1PK=OWNER#<owner>
                                                    GSI1SK=UPLOADED#<iso-ts>#<id>
    status event:  PK=DOC#<id>   SK=EVENT#<iso-ts>  (no GSI keys — sparse)
"""

from datetime import timedelta
from typing import Any

from boto3.dynamodb.types import TypeDeserializer, TypeSerializer

from app.models import (
    VALID_TRANSITIONS,
    DocumentRecord,
    DocumentStatus,
    StatusEvent,
    utcnow,
)

_ser = TypeSerializer()
_de = TypeDeserializer()

EVENT_TTL = timedelta(days=7)   # status events clean themselves up


def _marshal(data: dict[str, Any]) -> dict[str, Any]:
    """Python dict -> DynamoDB wire format. NB: floats are rejected —
    DynamoDB numbers are Decimal; our numeric fields are ints, which is fine."""
    return {k: _ser.serialize(v) for k, v in data.items()}


def _unmarshal(item: dict[str, Any]) -> dict[str, Any]:
    return {k: _de.deserialize(v) for k, v in item.items()}


class StatusConflictError(Exception):
    """The document was not in the expected status (a lost race, or an
    illegal transition). The transaction was cancelled; nothing changed."""


class DocumentRepository:
    def __init__(self, client: Any, table_name: str) -> None:
        self._c = client
        self._table = table_name

    # -- item shapes ---------------------------------------------------

    @staticmethod
    def _document_item(record: DocumentRecord) -> dict[str, Any]:
        return {
            "PK": f"DOC#{record.document_id}",
            "SK": "META",
            "entity_type": "document",
            "GSI1PK": f"OWNER#{record.owner}",
            # ISO-8601 UTC sorts lexicographically == chronologically;
            # the id suffix keeps ordering stable within one timestamp.
            "GSI1SK": f"UPLOADED#{record.uploaded_at.isoformat()}#{record.document_id}",
            **record.model_dump(mode="json", exclude_none=True),
        }

    # -- pattern 1: get by id -------------------------------------------

    async def get(self, document_id: str) -> DocumentRecord | None:
        resp = await self._c.get_item(
            TableName=self._table,
            Key=_marshal({"PK": f"DOC#{document_id}", "SK": "META"}),
        )
        if "Item" not in resp:
            return None
        # Extra attributes (PK, SK, GSI1*, entity_type) are ignored by
        # pydantic; Decimals coerce back into the declared int fields.
        return DocumentRecord.model_validate(_unmarshal(resp["Item"]))

    # -- create ----------------------------------------------------------

    async def create(self, record: DocumentRecord) -> None:
        await self._c.put_item(
            TableName=self._table,
            Item=_marshal(self._document_item(record)),
            # A fresh UUID shouldn't collide, but the condition costs
            # nothing and makes an overwrite impossible rather than unlikely.
            ConditionExpression="attribute_not_exists(PK)",
        )

    # -- pattern 2: list by owner, newest first (GSI1) --------------------

    async def list_by_owner(
        self,
        owner: str,
        limit: int = 20,
        start_key: dict[str, Any] | None = None,
    ) -> tuple[list[DocumentRecord], dict[str, Any] | None]:
        kwargs: dict[str, Any] = {
            "TableName": self._table,
            "IndexName": "GSI1",
            "KeyConditionExpression": "GSI1PK = :pk",
            "ExpressionAttributeValues": _marshal({":pk": f"OWNER#{owner}"}),
            "ScanIndexForward": False,   # descending sort key = newest first
            "Limit": limit,
        }
        if start_key:
            kwargs["ExclusiveStartKey"] = start_key
        resp = await self._c.query(**kwargs)
        records = [DocumentRecord.model_validate(_unmarshal(i)) for i in resp["Items"]]
        return records, resp.get("LastEvaluatedKey")

    # -- pattern 3: status history (one item collection, one query) -------

    async def history(self, document_id: str) -> list[StatusEvent]:
        resp = await self._c.query(
            TableName=self._table,
            KeyConditionExpression="PK = :pk AND begins_with(SK, :prefix)",
            ExpressionAttributeValues=_marshal(
                {":pk": f"DOC#{document_id}", ":prefix": "EVENT#"}
            ),
        )
        return [StatusEvent.model_validate(_unmarshal(i)) for i in resp["Items"]]

    # -- pattern 4: the transactional status update -----------------------

    async def update_status(
        self,
        document_id: str,
        to_status: DocumentStatus,
        expected: DocumentStatus,
        detail: str = "",
    ) -> None:
        """Move a document through the state machine, atomically writing
        the audit event with it. Raises StatusConflictError on a lost race
        or an illegal transition — in either case, nothing was written."""
        if to_status not in VALID_TRANSITIONS[expected]:
            raise StatusConflictError(f"illegal transition {expected} -> {to_status}")

        now = utcnow()
        try:
            await self._c.transact_write_items(
                TransactItems=[
                    {
                        "Update": {
                            "TableName": self._table,
                            "Key": _marshal({"PK": f"DOC#{document_id}", "SK": "META"}),
                            # 'status' is a DynamoDB reserved word, hence #s.
                            "UpdateExpression": "SET #s = :new, updated_at = :now",
                            "ConditionExpression": "attribute_exists(PK) AND #s = :expected",
                            "ExpressionAttributeNames": {"#s": "status"},
                            "ExpressionAttributeValues": _marshal(
                                {
                                    ":new": to_status.value,
                                    ":expected": expected.value,
                                    ":now": now.isoformat(),
                                }
                            ),
                        }
                    },
                    {
                        "Put": {
                            "TableName": self._table,
                            "Item": _marshal(
                                {
                                    "PK": f"DOC#{document_id}",
                                    "SK": f"EVENT#{now.isoformat()}",
                                    "entity_type": "status_event",
                                    "document_id": document_id,
                                    "occurred_at": now.isoformat(),
                                    "from_status": expected.value,
                                    "to_status": to_status.value,
                                    "detail": detail,
                                    # TTL attribute: epoch seconds.
                                    "expires_at": int((now + EVENT_TTL).timestamp()),
                                }
                            ),
                        }
                    },
                ]
            )
        except self._c.exceptions.TransactionCanceledException as exc:
            raise StatusConflictError(
                f"document {document_id} was not in status '{expected.value}'"
            ) from exc
```

A few details are worth calling out before wiring it up:

- **`status` is a reserved word.** DynamoDB has a [long list](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ReservedWords.html) of them, and `status`, `owner`, `language`, and `size` are all on it. Storing such attributes is fine; it is expressions that reject them, which is why reserved names go through `ExpressionAttributeNames` placeholders (`#s`). If you write `SET status = :new` directly, you get a `ValidationException` which at least quotes the offending token, so it is easy to diagnose.
- **The serializer rejects `float`.** DynamoDB numbers are decimals, and `TypeSerializer` raises on a Python float rather than silently losing precision. Our numeric fields are all `int`, so this does not come up here. It will if Part 7 ever adds a `confidence: float` field; the fix then is to store a `Decimal`.
- **The condition expression is the concurrency control.** `update_status` does not trust any previous read. It asserts the expected current status inside the write, so two workers both trying `uploaded → processing` cannot both succeed, no matter how they interleave. The losing transaction is cancelled entirely, together with its audit event.

### 5. Wiring it into the app

The lifespan already holds an `AsyncExitStack` from Part 3, so opening the DynamoDB client is one more line, and the factory from Part 3 needs no changes (`app/main.py`):

```python
from app.models import DocumentRecord, DocumentStatus
from app.repository import DocumentRepository, StatusConflictError


@asynccontextmanager
async def lifespan(app: FastAPI):
    cache = ConfigCache(ttl_seconds=get_settings().config_ttl_seconds)
    cache.get()
    app.state.config_cache = cache

    async with AsyncExitStack() as stack:
        app.state.s3 = await open_async_client(stack, "s3")
        app.state.dynamodb = await open_async_client(stack, "dynamodb")   # NEW
        yield


def get_repository(config: AppConfig = Depends(get_app_config)) -> DocumentRepository:
    """FastAPI dependency: a repository bound to the live client + table name."""
    return DocumentRepository(app.state.dynamodb, config.table_name)
```

The upload endpoint now writes a metadata record after putting the file into S3. One placeholder is needed here: documents need an *owner*, but authentication does not arrive until Part 8, so for now the owner comes from an `X-Owner` header with a demo default. Treat this as scaffolding, not as a security design; Part 8 replaces it with real STS-backed identity.

```python
from fastapi import Header


@app.post("/documents", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    x_owner: str = Header("demo@example.com", alias="X-Owner"),
    config: AppConfig = Depends(get_app_config),
    repo: DocumentRepository = Depends(get_repository),
) -> DocumentRecord:
    record = DocumentRecord(
        document_id=str(uuid.uuid4()),
        owner=x_owner,
        filename=file.filename or "unnamed",
        content_type=file.content_type or "application/octet-stream",
        size=file.size or 0,
    )

    # S3 keeps the bytes (Part 3); DynamoDB keeps what we know about them.
    await app.state.s3.upload_fileobj(
        file.file,
        config.bucket_name,
        record.document_id,
        ExtraArgs={"ContentType": record.content_type},
    )
    await repo.create(record)
    return record
```

(The `Metadata={"filename": ...}` trick from Part 3 is gone; the filename lives in the record now, as promised.)

Now the Part 3 placeholder finally gets replaced. `GET /documents` becomes a real query against `GSI1`, with real pagination: `LastEvaluatedKey` round-trips to the client as an opaque base64 token:

```python
import base64
import json


def _encode_token(key: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(key).encode()).decode()


def _decode_token(token: str) -> dict:
    try:
        return json.loads(base64.urlsafe_b64decode(token))
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="invalid next_token") from exc


@app.get("/documents")
async def list_documents(
    x_owner: str = Header("demo@example.com", alias="X-Owner"),
    limit: int = 20,
    next_token: str | None = None,
    repo: DocumentRepository = Depends(get_repository),
) -> dict:
    start_key = _decode_token(next_token) if next_token else None
    records, last_key = await repo.list_by_owner(
        x_owner, limit=limit, start_key=start_key
    )
    return {
        "documents": records,
        "next_token": _encode_token(last_key) if last_key else None,
    }


@app.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    repo: DocumentRepository = Depends(get_repository),
) -> DocumentRecord:
    record = await repo.get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    return record


@app.get("/documents/{document_id}/history")
async def get_document_history(
    document_id: str,
    repo: DocumentRepository = Depends(get_repository),
) -> dict:
    return {"events": await repo.history(document_id)}
```

Finally, we add a route to exercise the transaction by hand. In the finished product, status changes come from the worker (Part 5) and the Lambda (Part 7); no user-facing API would offer this. So it is a **debug route**, guarded like `/config` from Part 2, which lets us play the worker's part manually until the worker exists:

```python
from pydantic import BaseModel as RequestModel


class StatusChange(RequestModel):
    to: DocumentStatus
    detail: str = ""


@app.post("/documents/{document_id}/status")
async def change_status(
    document_id: str,
    change: StatusChange,
    repo: DocumentRepository = Depends(get_repository),
) -> DocumentRecord:
    """Simulate the worker's status transition. Debug-only until Part 5."""
    if not get_settings().debug_routes:
        raise HTTPException(status_code=404)

    record = await repo.get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")

    try:
        # The read above told us the expected 'from' status; the condition
        # inside the transaction re-asserts it, so a race loses cleanly.
        await repo.update_status(
            document_id, change.to, expected=record.status, detail=change.detail
        )
    except StatusConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return await repo.get(document_id)
```

### 6. The full loop

```bash
make up      # MiniStack
make seed    # params + secret + bucket + table/GSI/Streams/TTL (this part)
make run
```

Upload as two different owners:

```bash
curl -s -X POST localhost:8000/documents -H "X-Owner: alice@example.com" \
     -F "file=@contract.pdf;type=application/pdf" | jq '{document_id, owner, status}'
# {
#   "document_id": "8c2f6a1e-...",
#   "owner": "alice@example.com",
#   "status": "uploaded"
# }

curl -s -X POST localhost:8000/documents -H "X-Owner: bob@example.com" \
     -F "file=@notes.txt;type=text/plain" | jq .document_id
# "d41a77b0-..."
```

The list is now owner-scoped, so Alice does not see Bob's document:

```bash
curl -s localhost:8000/documents -H "X-Owner: alice@example.com" \
    | jq '.documents[] | {filename, status, uploaded_at}'
# {
#   "filename": "contract.pdf",
#   "status": "uploaded",
#   "uploaded_at": "2026-08-01T09:15:42.114532+00:00"
# }
```

Walk the document through its state machine, then attempt an illegal transition:

```bash
DOC=8c2f6a1e-...

curl -s -X POST localhost:8000/documents/$DOC/status \
     -H 'Content-Type: application/json' -d '{"to": "processing"}' | jq .status
# "processing"

curl -s -X POST localhost:8000/documents/$DOC/status \
     -H 'Content-Type: application/json' \
     -d '{"to": "processed", "detail": "simulated by hand"}' | jq .status
# "processed"

# 'processed' is terminal — this transition is illegal:
curl -s -X POST localhost:8000/documents/$DOC/status \
     -H 'Content-Type: application/json' -d '{"to": "processing"}' | jq .
# {
#   "detail": "illegal transition DocumentStatus.PROCESSED -> DocumentStatus.PROCESSING"
# }
```

And the audit trail the transaction has been writing all along (pattern 3, one query over one item collection):

```bash
curl -s localhost:8000/documents/$DOC/history | jq '.events[] | {from_status, to_status}'
# { "from_status": "uploaded",   "to_status": "processing" }
# { "from_status": "processing", "to_status": "processed" }
```

Each of those events carries an `expires_at` value seven days in the future, so the trail cleans itself up without any extra code.

### 7. Side-quest: watching the stream

The table has been recording every one of those writes into its stream, so let us take a look. `bootstrap/tail_stream.py` below is deliberately simple: it uses synchronous `boto3` and assumes one shard. That is fine locally; real streams shard and re-shard under load, which is much of the bookkeeping that Streams→Lambda event-source mappings exist to hide.

```python
"""Side-quest: tail the docinbox table's DynamoDB Stream.

Run with:  python -m bootstrap.tail_stream   (Ctrl+C to stop)
"""

import time

from app.aws.clients import get_client
from bootstrap.seed import PARAMETERS


def main() -> None:
    table = PARAMETERS["dynamodb/table-name"]
    arn = get_client("dynamodb").describe_table(TableName=table)["Table"]["LatestStreamArn"]

    streams = get_client("dynamodbstreams")
    shard = streams.describe_stream(StreamArn=arn)["StreamDescription"]["Shards"][0]
    iterator = streams.get_shard_iterator(
        StreamArn=arn, ShardId=shard["ShardId"], ShardIteratorType="LATEST"
    )["ShardIterator"]

    print(f"tailing {arn}\n")
    while True:
        resp = streams.get_records(ShardIterator=iterator, Limit=25)
        for rec in resp["Records"]:
            keys = rec["dynamodb"]["Keys"]          # raw wire format: {"S": ...}
            print(f'{rec["eventName"]:<7} {keys["PK"]["S"]:<45} {keys["SK"]["S"]}')
        iterator = resp["NextShardIterator"]
        time.sleep(1)


if __name__ == "__main__":
    main()
```

Run it in a second terminal, then upload a document and push it through a status change in the first:

```
INSERT  DOC#f7a3c9d2-...                              META
MODIFY  DOC#f7a3c9d2-...                              META
INSERT  DOC#f7a3c9d2-...                              EVENT#2026-08-01T09:31:07...
```

Those three records show the upload (INSERT of `META`), followed by the transaction (MODIFY of `META` plus INSERT of the event, arriving together). Also worth noting is what the stream does *not* say: nothing here reads "a document wants processing", and when the TTL sweep eventually deletes an expired event, its REMOVE record will look like any other delete. That is the gap described in the Concepts section, and it is why Part 5 uses explicit SNS events instead.

---

## The general pattern

> **The portable version, if you are building something else:** write down every question your application asks of its data *before* designing keys, then give each question a key shape: `GetItem` for identity lookups, a `Query` over one item collection for lists, and a GSI when the same data must be answered by a second dimension. Keep every key decision in one repository module that speaks your domain types, so the table layout stays invisible above it. Enforce state machines with condition expressions inside the write (never read-check-write), and use a transaction whenever a state change and its side record must succeed or fail together. Finally, give short-lived operational records a TTL when you create them, while it is still one attribute instead of a cleanup project.
{: .prompt-tip }

## In production

> - **On-demand billing (`PAY_PER_REQUEST`) is the sensible default** until there is enough traffic to model. Provisioned capacity with auto-scaling can be cheaper at sustained load, but it introduces throttling (`ProvisionedThroughputExceededException`), and MiniStack will never throw that error at you, so the retry path stays untested until real AWS.
> - GSI eventual consistency is real there: an upload followed immediately by a list can miss the new document. UIs normally handle this by echoing the created record into the view rather than re-querying; do not paper over it with a sleep.
> - **The 400 KB item limit is real too.** Our design already follows the standard answer: big payloads (document bytes, extracted text) live in S3, and DynamoDB holds pointers plus queryable metadata. If the Part 7 summaries ever grow too large, they move to S3 the same way.
> - Single-table design is not an analytics format, and that is expected; it is optimized for the application's questions. The usual answer is to export (or stream) the table into S3 and point Athena at the copy, rather than running `Scan`s against the production table.
> - Table creation belongs to Terraform/CloudFormation, same as the bucket configuration in Part 3. The seed script is a local convenience only.
{: .prompt-info }

## MiniStack notes and fidelity

- **The core feature set we used is all emulated**: GSIs, condition expressions, `TransactWriteItems` (including cancellation on failed conditions), TTL, Streams, pagination via `LastEvaluatedKey`, and the reserved-word rules. The whole repository runs unchanged against real AWS (endpoint unset, as always).
- **TTL sweep cadence differs.** MiniStack sweeps on a short local cadence, so an expired event disappears within a minute or two, which is convenient for the exercise below. Real AWS promises only "typically within a few days" after expiry. Both environments agree on the important part: expired-but-unswept items still appear in queries, so filter them if it matters.
- **GSI reads are instantly consistent locally.** MiniStack applies index updates synchronously, so the write-then-miss window that real GSIs have simply does not exist here. This is the fidelity gap most likely to bite this part's code in production; it is on the list for the parity checks in Part 10.
- **No throughput, throttling, or item-size enforcement.** You will not see a throttle or a 400 KB rejection locally, which also means you will not notice you are heading toward one.
- **Streams fidelity is good for learning, but simplified in shape.** Expect a single shard locally, where real tables shard by partition and re-shard under load. The record format (`eventName`, `Keys`, `NewImage`/`OldImage`) matches the real service, and Streams→Lambda event-source mappings work if you want to wire one up after Part 7.

## Troubleshooting

- **`ResourceNotFoundException` on any repository call.** The table does not exist; `make seed` was not run after pulling this part. Verify with `aws --endpoint-url http://localhost:4566 dynamodb list-tables`.
- **`ValidationException` complaining about a reserved keyword.** You wrote `status`, `owner`, `size`, or `language` directly in an expression. Route it through `ExpressionAttributeNames` (see `#s` in `update_status`).
- **`TypeError: Float types are not supported. Use Decimal types instead.`** Something put a Python `float` into an item. Convert to `Decimal` (or `int`/`str`) before it reaches the serializer. Do not "fix" this by rounding to int unless the value really is one.
- **`TransactionCanceledException` when calling the API by hand.** That is the condition doing its job: the document was not in the status you claimed. If you need to know *which* clause failed, the exception's `CancellationReasons` lists one entry per transact item, in order.
- **`GET /documents` returns empty but uploads succeed.** Check the `X-Owner` header on the list call; with no header, you are listing `demo@example.com`'s documents, and the upload probably went to a different owner. (The placeholder identity is exactly this flimsy; Part 8 fixes it.)
- **`POST /documents/{id}/status` returns 404 for a document that exists.** `DEBUG_ROUTES` is off; the guard 404s rather than advertising the route, same as `/config` in Part 2.

## Weekend exercises

1. **Add a second access pattern via a new GSI.** "List all documents in status `failed`" is what an ops dashboard would ask. Add `GSI2PK = STATUS#<status>`, `GSI2SK = UPLOADED#<iso-ts>#<id>` to the document item, create `GSI2` in the seed script (note: adding a GSI to an *existing* table is `update_table`, and the seed script needs to stay idempotent around it), and update the two places the repository writes those attributes: creation, and `update_status`, which now must keep `GSI2PK` in sync.
2. **Write a transaction that touches two items.** Add a per-owner counter item (`PK=OWNER#<owner>`, `SK=COUNT`) and extend `create()` into a transaction: put the document *and* `ADD document_count :one` in one unit. Then upload concurrently (`for i in $(seq 10); do curl ... & done; wait`) and confirm the count is exact; that is the property the transaction bought you.
3. **Page through results.** Set `limit=2`, upload five documents as one owner, and follow `next_token` until it comes back `null`. Then try tampering with a token (flip one character) and watch the 400. Worth noting: a token encodes a position in the index, so there is no "jump to page 7" in DynamoDB. Product UIs built on it tend to use infinite scroll for exactly this reason.
4. **Watch TTL actually fire.** Drop `EVENT_TTL` to `timedelta(seconds=30)`, make a status change, and run the stream tailer until the REMOVE record for the event scrolls by. Locally that takes a couple of minutes at most (see the fidelity notes); in real AWS it could be days, which is why code that cares has to filter on `expires_at` itself.

## What is next

`docinbox` now has a memory. Documents belong to owners, listing is a real query with real pagination, and every document carries a status that can only change along legal transitions. The schema is already shaped for the LLM fields that arrive in Part 7. However, nothing *reacts* to an upload yet; the document just sits in `uploaded` until someone calls the debug route by hand.

**Part 5 — Async Messaging I: SNS + SQS.** In the next part, the upload endpoint starts publishing a `document.uploaded` event to an SNS topic, an SQS queue subscribes to it, and a standalone worker long-polls the queue and drives the status transitions we built in this part: the same `update_status`, called by a real consumer instead of curl. `docker-compose` gains its first worker service, and the pipeline starts to look like the architecture diagram.

The source code used in this part is available in the companion GitHub repository: [github.com/chuan2019/docinbox](https://github.com/chuan2019/docinbox) (tag `part-04`).

*See you next weekend.*
