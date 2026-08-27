# Quant Research Radar

A provenance-first research and learning system for a beginner quantitative researcher. It collects public research and read-only market observations, preserves raw evidence, extracts explicitly classified claims, generates falsifiable hypothesis candidates, and teaches the concepts needed to evaluate them.

**This is not a trading bot or an AI signal service.** It does not place orders, manage portfolios, hold wallets, or produce trading recommendations.

## Epistemic model

- **FACT**: an observation directly supported by preserved source evidence.
- **CLAIM**: a statement reported by a source, not automatically true.
- **HYPOTHESIS**: a falsifiable idea that remains untested until independently evaluated.

`CLAIM != FACT` and `HYPOTHESIS != ALPHA`. LLM interpretations are derived analysis and never overwrite raw source evidence. Missing evidence fails closed: values and timestamps are not invented or forward-filled.

## Architecture

`Sources -> Raw Evidence -> Claims -> Hypotheses -> Reports`

The MVP uses SQLAlchemy models with PostgreSQL as the intended persistent backend; SQLite is convenient for offline tests. Adapters are bounded and idempotent. The default Fake LLM makes the pipeline deterministic without an API key.

Initial sources:

- arXiv official Atom API
- RePEc/IDEAS NEP RSS endpoint (with clean degraded behavior if unavailable)
- Hyperliquid public read-only info endpoint, initially BTC/ETH/SOL

## Setup

Requires Python 3.11+. With uv:

```bash
uv sync --extra dev
cp .env.example .env
quant-radar init-db
```

Set `DATABASE_URL` to PostgreSQL for normal development, or use a SQLite URL for local/offline work. Run migrations with:

```bash
alembic upgrade head
```

The `.env.example` file contains conservative HTTP, source, model, and collection bounds. Never commit `.env` or API keys.

## CLI

```bash
quant-radar collect arxiv --offline
quant-radar collect repec --offline
quant-radar collect hyperliquid --offline
quant-radar analyze
quant-radar report daily
quant-radar report weekly
quant-radar run-daily --offline
```

`run-daily` performs one bounded invocation and exits. It generates `outputs/daily-YYYY-MM-DD.md`.

## Research workflow

`Observation -> Mechanism -> Testable Hypothesis -> Required Data -> Potential Biases -> Empirical Test`

The longer-term workflow is `Observation -> Hypothesis -> Event Study -> Backtest -> Walk-forward -> Paper`. Event studies, backtests, walk-forward analysis, paper trading, and any capital deployment are later phases and are not implemented here.

## Scoring

Hypotheses receive transparent deterministic components: economic mechanism (20), evidence quality (15), data availability (20), executability (20), alpha half-life suitability (15), and simplicity (10), with explicit penalties. The LLM cannot assign the final score.

## Quality gates

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

## Explicit non-goals

No trading, execution, wallets, signing, order placement, BUY/SELL/LONG/SHORT signals, X/Twitter, Reddit, SSRN scraping, Dune, Kalshi, Polymarket monitoring, Binance, DeFi, options, dashboards, React, mobile, Kafka, Redis, Kubernetes, vector databases, embeddings, agent swarms, or production trading infrastructure.
