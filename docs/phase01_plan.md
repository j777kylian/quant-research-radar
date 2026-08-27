# Phase 0/1 Implementation Plan — Quant Research Radar

Date: 2026-08-27

## Purpose

`quant-research-radar` is a **research and learning system** for a beginner
quantitative researcher. It continuously collects public research and market
data, extracts observations and claims, transforms interesting ideas into
falsifiable quantitative hypotheses, teaches the concepts needed to understand
them, and maintains a durable Alpha/Hypothesis Library.

It is **not** a trading bot and produces **no** trading signals, orders, or
capital-deployment logic.

Long-term pipeline (later phases, not implemented here):

```
Sources -> Raw Evidence -> Claims -> Hypotheses -> Research Queue
        -> Event Study -> Backtest -> Walk-forward -> Paper Trading
```

This phase implements everything up to and including *Reports*:

```
Sources -> Raw Evidence -> Claims -> Hypotheses -> Reports
```

## Epistemic model

The system enforces five rules (see README for full text):

1. `CLAIM != FACT` — statements from papers/posts/models are classified, never
   silently promoted.
2. `HYPOTHESIS != ALPHA` — a hypothesis is untested until independently
   evaluated.
3. LLM inference never silently becomes source evidence. Source text, extracted
   observation, claim, LLM interpretation, and generated hypothesis are distinct
   database concepts.
4. Every analytical object retains provenance (source, URL, source ID, author,
   timestamps, raw payload, content hash, model used, analysis timestamp).
5. Research fails closed: never invent, forward-fill, or infer missing values.

## Architecture

```
config (pydantic-settings, .env)
  -> sources (adapters: arXiv, RePEc, Hyperliquid)
  -> collectors (idempotent upsert + observability CollectionRun)
  -> db (SQLAlchemy 2.x models, Alembic)
  -> llm (LLMClient interface: fake deterministic + optional real provider;
          pydantic-validated structured outputs)
  -> hypotheses (transparent scoring, library)
  -> market (derived metrics from raw observations)
  -> pipeline (bounded daily / weekly orchestration)
  -> reports (Markdown Daily / Weekly generators)
  -> cli (argparse, `quant-radar ...`)
```

Data flow per run:

```
collect(source) -> source_items (raw, hashed, idempotent)
  -> triage (retain/archive + scores)            [LLM Role 1]
  -> analyst (claims + hypothesis candidates)     [LLM Role 2]
  -> critic (failure modes, biases)               [LLM Role 3]
  -> scoring (transparent component score)        [deterministic, not LLM]
  -> tutor (1-2 concepts)                         [LLM Role 4]
  -> reports (daily / weekly Markdown)
```

## Data model

| Table | Purpose |
| --- | --- |
| `source_items` | Raw externally retrieved records; provenance + SHA-256; unique `(source_type, external_id)` |
| `claims` | Extracted statements with explicit classification `FACT/CLAIM/OPINION/RESULT` |
| `hypotheses` | Falsifiable research hypotheses with full lifecycle/status |
| `concepts` | Educational material (beginner/technical explanation, formula, example) |
| `reviews` | Research evaluations: `DAILY/WEEKLY/MANUAL` x `IGNORE/WATCH/RESEARCH/TEST` |
| `market_observations` | Raw point-in-time Hyperliquid funding data (BTC/ETH/SOL) |
| `market_metrics` | Derived metrics (percentiles, changes, returns, volatility) with provenance |
| `collection_runs` | Per-collection observability (requested/retrieved/inserted/updated/skipped/failed) |
| `analysis_runs` | Per-LLM-run observability (provider, model, schema version, counts) |

Hypothesis lifecycle: `DISCOVERED -> FORMALIZED -> DATA_AVAILABLE -> TEST_READY
-> TESTING -> {REJECTED | PROMISING -> BACKTESTED -> WALK_FORWARD -> PAPER}`.
Rejected hypotheses are retained — failure is research information.

### Database/backend compatibility

Production backend is PostgreSQL. Tests run on SQLite (in-memory) so the normal
suite is offline and deterministic. To keep semantics compatible:

- JSON columns use the generic `sa.JSON` type (JSONB on PG is not required).
- UUID PKs use `sa.Uuid` (native UUID on PG, CHAR(32) on SQLite).
- Timestamps use a `UTCDateTime` type decorator so values are always
  timezone-aware UTC on both backends.
- No server-side-only defaults; defaults are Python-side.

## Source adapters

All adapters implement a common `SourceAdapter` interface returning normalized
records; collectors handle idempotency, hashing, and observability.

| Source | Interface | Method | Degraded mode |
| --- | --- | --- | --- |
| arXiv | Official Atom API (`export.arxiv.org/api/query`), topic-filtered query | httpx GET, XML parse | none needed |
| RePEc | Stable public feed (IDEAS/NEP RSS or documented fallback) | httpx GET, RSS/XML parse | documented manual/offline import; collector degrades cleanly |
| Hyperliquid | Public read-only `POST api.hyperliquid.xyz/info` | httpx POST, JSON | live funding history; candle snapshot for returns |

Bounded collection: configurable lookback + `--limit`; no full-history backfills.

## LLM architecture

- `LLMClient` interface with two implementations:
  - `FakeLLMClient` — deterministic, offline, content-derived outputs (default).
  - `OpenAICompatClient` — optional real provider via env config, structured
    JSON outputs validated by Pydantic before persistence.
- Four roles with dedicated Pydantic schemas: `Triage`, `Analyst`, `Critic`,
  `Tutor`.
- A real LLM key is never required for tests or the offline daily run.
- Raw source text is never overwritten by LLM output (`RAW SOURCE` /
  `DERIVED ANALYSIS` / `RESEARCH HYPOTHESIS` stay separate in code and DB).

## Hypothesis scoring

Transparent, component-based, computed deterministically in code (not invented
by the LLM):

- Economic mechanism 0-20, Evidence quality 0-15, Data availability 0-20,
  Executability 0-20, Alpha half-life 0-15, Simplicity 0-10 → max 100.
- Explicit penalties (crowded, latency-sensitive, weak PIT data, transaction
  costs, unresolved provenance).
- Event-study eligibility: score >= 70 + data obtainable + PIT-safe design +
  executable by a non-institutional researcher + falsifiable.
  (Event Study itself is a later phase.)

## CLI

```
quant-radar init-db
quant-radar collect arxiv|repec|hyperliquid [--limit N] [--offline]
quant-radar analyze [--limit N] [--offline]        # triage+hypotheses+critic+tutor
quant-radar report daily [--date YYYY-MM-DD]
quant-radar report weekly [--end YYYY-MM-DD]
quant-radar run-daily [--limit N] [--offline]      # one bounded end-to-end run
```

One invocation is bounded and exits; scheduling is left to cron/systemd.

## Test strategy

Offline, deterministic unit/integration tests against SQLite:

- Collectors: parsing, idempotent re-ingestion, malformed responses, timeouts,
  duplicate handling.
- DB: constraints, hypothesis lifecycle transitions, claim classifications,
  provenance relations.
- LLM: schema validation, malformed-output rejection, fake deterministic runs.
- Reports: deterministic structure, max item counts, FACT/CLAIM/HYPOTHESIS
  labeling, no trading commands.
- Pipeline: bounded execution, one collector failing does not corrupt unrelated
  data, deterministic fixture run.

Live-API smoke checks are separate manual commands, never part of `pytest`.

## Quality gates

```
ruff check .
ruff format --check .
mypy src
pytest
```

## Explicit non-goals (Phase 0/1)

No trading, order placement, wallets, signing, portfolio management,
BUY/SELL/LONG/SHORT signals, X/Twitter, Reddit, SSRN scraping, Dune, Kalshi,
Polymarket, Binance, DeFi protocols, options, GitHub discovery, web dashboard,
React, mobile, Kafka, Redis, Kubernetes, vector DB, embeddings, agent swarms,
autonomous trading, or production trading infrastructure.

## Development sequence

Bootstrap → DB schema → source collectors → LLM schemas → pipeline/reporting →
CLI → tests → quality gates → README → sample report. Coherent git commits at
each stage; no pushes.
