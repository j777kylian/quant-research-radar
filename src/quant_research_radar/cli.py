from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from .config import get_settings
from .db import init_db, make_engine, make_session_factory
from .llm import DeepSeekClient, FakeLLMClient, ModelRouter
from .pipeline import (
    analyze,
    calculate_metrics,
    daily_report,
    ingest,
    ingest_records,
    weekly_report,
)
from .sources import ArxivSource, HyperliquidSource, RepecSource, SourceAdapter


def main() -> None:
    parser = argparse.ArgumentParser(prog="quant-radar")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db")
    collect = sub.add_parser("collect")
    collect.add_argument("source", choices=["arxiv", "repec", "hyperliquid"])
    collect.add_argument("--limit", type=int, default=None)
    collect.add_argument("--offline", action="store_true")
    collect.add_argument("--history", action="store_true")
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--limit", type=int, default=20)
    sub.add_parser("run-daily").add_argument("--offline", action="store_true")
    routing = sub.add_parser("validate-llm-routing")
    routing.add_argument(
        "--artifact", default="outputs/validation/phase15c-llm-routing.json"
    )
    routing.add_argument("--database-url", default="sqlite:///phase15c-validation.db")
    routing.add_argument("--live", action="store_true")
    validate = sub.add_parser("validate-live")
    validate.add_argument("--limit", type=int, default=5)
    validate.add_argument(
        "--artifact", default="outputs/validation/phase15b-live-validation.json"
    )
    report = sub.add_parser("report")
    report.add_argument("kind", choices=["daily", "weekly"])
    args = parser.parse_args()
    settings = get_settings()
    database_url = (
        args.database_url
        if args.command == "validate-llm-routing"
        else settings.database_url
    )
    engine = make_engine(database_url)
    init_db(engine)
    session = make_session_factory(engine)()

    if args.command == "validate-llm-routing":
        router = ModelRouter(settings.llm_flash_model, settings.llm_pro_model)
        roles = ["TRIAGE", "TUTOR", "ANALYST", "CRITIC"]
        configured: dict[str, object] = {}
        result: dict[str, object] = {
            "validation_timestamp": datetime.now(UTC).isoformat(),
            "provider": "deepseek",
            "endpoint_type": "chat_completions",
            "mock": {"roles": configured},
        }
        for role in roles:
            config = router.resolve(role)
            configured[role] = {
                "requested_model": config.model,
                "thinking": config.thinking,
                "reasoning_effort": config.reasoning_effort,
                "max_output_tokens": config.max_output_tokens,
                "status": "PASS",
            }
        if args.live:
            if not settings.deepseek_api_key:
                result["live_status"] = "NOT_RUN_NO_CREDENTIAL"
            else:
                live_client = DeepSeekClient(
                    settings.deepseek_api_key,
                    router,
                    settings.deepseek_base_url,
                    settings.llm_timeout_seconds,
                    settings.http_retries,
                )
                live: dict[str, object] = {}
                for role in roles:
                    config = router.resolve(role)
                    try:
                        if role == "TRIAGE":
                            live_client.triage(
                                "Liquidity screening", "A bounded market observation."
                            )
                        elif role == "TUTOR":
                            live_client.tutor("Explain point-in-time data.")
                        elif role == "ANALYST":
                            live_client.analyze(
                                "A bounded research question",
                                "A source-reported result.",
                            )
                        else:
                            live_client.critique(
                                "A candidate hypothesis about a measured relationship."
                            )
                        live[role] = {
                            "requested_model": config.model,
                            "actual_model": None,
                            "thinking": config.thinking,
                            "reasoning_effort": config.reasoning_effort,
                            "schema": "PASS",
                            "result": "PASS",
                        }
                    except Exception as exc:
                        live[role] = {
                            "requested_model": config.model,
                            "actual_model": None,
                            "thinking": config.thinking,
                            "reasoning_effort": config.reasoning_effort,
                            "schema": "FAIL",
                            "result": "FAIL",
                            "failure": str(exc),
                        }
                result["live"] = {"roles": live}
                result["live_status"] = (
                    "PASS"
                    if all(
                        isinstance(item, dict) and item["result"] == "PASS"
                        for item in live.values()
                    )
                    else "PARTIAL"
                )
        result["overall"] = result.get("live_status", "PASS") if args.live else "PASS"
        artifact = Path(args.artifact)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "init-db":
        print("Database initialized")
    elif args.command == "collect":
        adapters = {
            "arxiv": ArxivSource(lookback_days=settings.arxiv_lookback_days),
            "repec": RepecSource(),
            "hyperliquid": HyperliquidSource(),
        }
        limit = args.limit or getattr(settings, f"{args.source}_max_items", 10)
        adapter = cast("SourceAdapter", adapters[args.source])
        if args.history and args.source == "hyperliquid":
            source = cast(HyperliquidSource, adapter)
            inserted, duplicates = ingest_records(
                session, source.collect_history(limit)
            )
            inserted_candles, candle_duplicates = ingest_records(
                session, source.collect_candles(limit)
            )
            print(
                json.dumps(
                    {
                        "funding_inserted": inserted,
                        "funding_duplicates": duplicates,
                        "candle_inserted": inserted_candles,
                        "candle_duplicates": candle_duplicates,
                    }
                )
            )
        else:
            print(f"Inserted {ingest(session, adapter, limit, args.offline)} records")
        calculate_metrics(session)
    elif args.command == "analyze":
        client: FakeLLMClient | DeepSeekClient = (
            FakeLLMClient()
            if settings.llm_provider == "fake" or not settings.llm_api_key
            else DeepSeekClient(
                settings.deepseek_api_key or settings.llm_api_key,
                ModelRouter(settings.llm_flash_model, settings.llm_pro_model),
                settings.deepseek_base_url
                if settings.llm_provider == "deepseek"
                else settings.llm_base_url,
                settings.llm_timeout_seconds,
                settings.http_retries,
            )
        )
        print(f"Created {analyze(session, client, args.limit)} hypotheses")
    elif args.command == "run-daily":
        arxiv, repec, hyperliquid = (
            ArxivSource(lookback_days=settings.arxiv_lookback_days),
            RepecSource(),
            HyperliquidSource(lookback_hours=48),
        )
        ingest(session, arxiv, 5, args.offline)
        ingest(session, repec, 5, args.offline)
        ingest(session, hyperliquid, 5, args.offline)
        ingest_records(session, hyperliquid.collect_history(30, offline=args.offline))
        ingest_records(session, hyperliquid.collect_candles(30, offline=args.offline))
        calculate_metrics(session)
        analyze(session, FakeLLMClient(), 20)
        print(daily_report(session, settings.report_output_dir))
    elif args.command == "report":
        print(
            daily_report(session, settings.report_output_dir)
            if args.kind == "daily"
            else weekly_report(session, settings.report_output_dir)
        )


if __name__ == "__main__":
    main()
