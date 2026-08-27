from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import httpx


@dataclass(frozen=True)
class SourceRecord:
    source_type: str
    source_name: str
    external_id: str
    title: str
    canonical_url: str | None
    authors: list[str]
    published_at: datetime | None
    raw_text: str
    raw_metadata: dict[str, Any]


class SourceAdapter(Protocol):
    name: str

    def collect(self, limit: int, offline: bool = False) -> list[SourceRecord]: ...


class ArxivSource:
    name = "arxiv"
    endpoint = "https://export.arxiv.org/api/query"

    def __init__(self, client: httpx.Client | None = None, lookback_days: int = 14):
        self.client = client or httpx.Client(
            timeout=20, headers={"User-Agent": "quant-research-radar/0.1"}
        )
        self.lookback_days = lookback_days

    def collect(self, limit: int, offline: bool = False) -> list[SourceRecord]:
        if offline:
            return [
                SourceRecord(
                    "ACADEMIC",
                    self.name,
                    "arxiv:offline-001",
                    "Funding-rate persistence in perpetual futures",
                    "https://arxiv.org/abs/offline-001",
                    ["Fixture Researcher"],
                    datetime.now(UTC) - timedelta(days=1),
                    "We study whether extreme funding rates predict subsequent returns.",
                    {"fixture": True},
                )
            ][:limit]
        response = self.client.get(
            self.endpoint,
            params={
                "search_query": "all:bitcoin OR all:cryptocurrency OR all:funding rate",
                "start": 0,
                "max_results": min(limit, 100),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        )
        response.raise_for_status()
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            raise ValueError("arXiv response was not valid XML") from exc
        ns = {"a": "http://www.w3.org/2005/Atom"}
        cutoff = datetime.now(UTC) - timedelta(days=self.lookback_days)
        records: list[SourceRecord] = []
        for entry in root.findall("a:entry", ns):
            published = _parse_date(entry.findtext("a:published", namespaces=ns))
            if published and published < cutoff:
                continue
            identifier = (entry.findtext("a:id", namespaces=ns) or "").strip()
            records.append(
                SourceRecord(
                    "ACADEMIC",
                    self.name,
                    identifier,
                    (entry.findtext("a:title", namespaces=ns) or "").strip(),
                    identifier,
                    [
                        (a.findtext("a:name", namespaces=ns) or "").strip()
                        for a in entry.findall("a:author", ns)
                    ],
                    published,
                    (entry.findtext("a:summary", namespaces=ns) or "").strip(),
                    {
                        "categories": [
                            c.attrib.get("term")
                            for c in entry.findall("a:category", ns)
                        ]
                    },
                )
            )
        return records[:limit]


class RepecSource:
    name = "repec"
    endpoint = "https://ideas.repec.org/n/nep-fmk.rdf"

    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=20)

    def collect(self, limit: int, offline: bool = False) -> list[SourceRecord]:
        if offline:
            return [
                SourceRecord(
                    "ACADEMIC",
                    self.name,
                    "repec:offline-001",
                    "Liquidity and return predictability",
                    "https://ideas.repec.org/p/offline/001",
                    ["Fixture Economist"],
                    datetime.now(UTC) - timedelta(days=2),
                    "This fixture reports an association between liquidity and future returns.",
                    {"fixture": True},
                )
            ][:limit]
        response = self.client.get(self.endpoint)
        response.raise_for_status()
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            raise ValueError("RePEc response was not valid XML") from exc
        records: list[SourceRecord] = []
        for item in root.findall(".//item")[:limit]:
            link = (item.findtext("link") or "").strip()
            external_id = link or (item.findtext("guid") or "").strip()
            if not external_id:
                continue
            records.append(
                SourceRecord(
                    "ACADEMIC",
                    self.name,
                    external_id,
                    (item.findtext("title") or "").strip(),
                    link or None,
                    [],
                    None,
                    (item.findtext("description") or "").strip(),
                    {"feed": self.endpoint, "degraded_mode": "RSS/RDF"},
                )
            )
        return records


class HyperliquidSource:
    name = "hyperliquid"
    endpoint = "https://api.hyperliquid.xyz/info"
    assets = ("BTC", "ETH", "SOL")

    def __init__(self, client: httpx.Client | None = None, lookback_hours: int = 24):
        self.client = client or httpx.Client(timeout=20)
        self.lookback_hours = lookback_hours

    def collect(self, limit: int, offline: bool = False) -> list[SourceRecord]:
        if offline:
            now = datetime.now(UTC).replace(second=0, microsecond=0)
            return [self._offline_record(asset, now) for asset in self.assets][:limit]
        payload = self._post({"type": "metaAndAssetCtxs"})
        universe, contexts = _meta_contexts(payload)
        now = datetime.now(UTC)
        records: list[SourceRecord] = []
        for asset in self.assets[:limit]:
            if asset not in universe:
                continue
            context = contexts[universe.index(asset)]
            records.append(
                SourceRecord(
                    "MARKET",
                    self.name,
                    f"snapshot:{asset}:{now.isoformat()}",
                    f"{asset} market snapshot",
                    None,
                    [],
                    now,
                    f"Hyperliquid {asset} point-in-time market snapshot",
                    {
                        "asset": asset,
                        "observation_timestamp": now.isoformat(),
                        "funding_rate": _number(context.get("funding")),
                        "mark_price": _number(context.get("markPx")),
                        "open_interest": _number(context.get("openInterest")),
                        "volume": _number(context.get("dayNtlVlm")),
                        "source_payload": context,
                    },
                )
            )
        return records

    def collect_history(
        self,
        limit: int,
        offline: bool = False,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[SourceRecord]:
        if offline:
            end = end or datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
            return [
                self._history_record(asset, end - timedelta(hours=i), i)
                for asset in self.assets
                for i in range(min(limit, 6))
                if end - timedelta(hours=i)
                >= (start or datetime.min.replace(tzinfo=UTC))
            ]
        end = end or datetime.now(UTC)
        end_ms = int(end.timestamp() * 1000)
        start_ms = int(
            (start or (end - timedelta(hours=self.lookback_hours))).timestamp() * 1000
        )
        records: list[SourceRecord] = []
        for asset in self.assets:
            funding = self._post(
                {
                    "type": "fundingHistory",
                    "coin": asset,
                    "startTime": start_ms,
                    "endTime": end_ms,
                }
            )
            for row in funding[:limit]:
                if "coin" not in row or str(row["coin"]) != asset:
                    continue
                timestamp = _timestamp(row.get("time"))
                rate = _number(row.get("fundingRate"))
                if timestamp is None or rate is None:
                    raise ValueError(f"Invalid Hyperliquid funding row for {asset}")
                if str(row.get("coin", asset)) != asset:
                    continue
                records.append(
                    SourceRecord(
                        "MARKET",
                        self.name,
                        f"funding:{asset}:{timestamp.isoformat()}",
                        f"{asset} funding event",
                        None,
                        [],
                        timestamp,
                        json.dumps(row, sort_keys=True),
                        {
                            "asset": asset,
                            "kind": "funding",
                            "funding_timestamp": timestamp.isoformat(),
                            "funding_rate": rate,
                            "source_payload": row,
                        },
                    )
                )
        return records

    def collect_candles(
        self,
        limit: int,
        interval: str = "1h",
        offline: bool = False,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[SourceRecord]:
        if offline:
            end = end or datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
            return [
                self._candle_record(asset, end - timedelta(hours=i), i)
                for asset in self.assets
                for i in range(min(limit, 30))
                if end - timedelta(hours=i)
                >= (start or datetime.min.replace(tzinfo=UTC))
            ]
        end = end or datetime.now(UTC)
        end_ms = int(end.timestamp() * 1000)
        start_ms = int(
            (start or (end - timedelta(hours=max(limit, 2)))).timestamp() * 1000
        )
        records: list[SourceRecord] = []
        for asset in self.assets:
            rows = self._post(
                {
                    "type": "candleSnapshot",
                    "req": {
                        "coin": asset,
                        "interval": interval,
                        "startTime": start_ms,
                        "endTime": end_ms,
                    },
                }
            )
            for row in rows[:limit]:
                timestamp = _timestamp(row.get("t"))
                close = _number(row.get("c"))
                if timestamp is None or close is None:
                    raise ValueError(f"Invalid Hyperliquid candle row for {asset}")
                records.append(
                    SourceRecord(
                        "MARKET",
                        self.name,
                        f"candle:{asset}:{interval}:{timestamp.isoformat()}",
                        f"{asset} {interval} candle",
                        None,
                        [],
                        timestamp,
                        json.dumps(row, sort_keys=True),
                        {
                            "asset": asset,
                            "kind": "candle",
                            "interval": interval,
                            "candle_close_timestamp": timestamp.isoformat(),
                            "close": close,
                            "source_payload": row,
                        },
                    )
                )
        return records

    def _post(self, payload: dict[str, Any]) -> Any:
        response = self.client.post(self.endpoint, json=payload)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _offline_record(asset: str, timestamp: datetime) -> SourceRecord:
        prices = {"BTC": 60000.0, "ETH": 3000.0, "SOL": 150.0}
        return SourceRecord(
            "MARKET",
            "hyperliquid",
            f"snapshot:{asset}:{timestamp.isoformat()}",
            f"{asset} market snapshot",
            None,
            [],
            timestamp,
            f"Funding observation for {asset}; raw fields are in metadata.",
            {
                "asset": asset,
                "observation_timestamp": timestamp.isoformat(),
                "funding_rate": 0.0001 if asset == "BTC" else 0.0002,
                "mark_price": prices[asset],
                "fixture": True,
            },
        )

    @staticmethod
    def _history_record(asset: str, timestamp: datetime, index: int) -> SourceRecord:
        rate = (index + 1) / 10000
        row = {
            "coin": asset,
            "time": int(timestamp.timestamp() * 1000),
            "fundingRate": str(rate),
        }
        return SourceRecord(
            "MARKET",
            "hyperliquid",
            f"funding:{asset}:{timestamp.isoformat()}",
            f"{asset} funding event",
            None,
            [],
            timestamp,
            json.dumps(row),
            {
                "asset": asset,
                "kind": "funding",
                "funding_timestamp": timestamp.isoformat(),
                "funding_rate": rate,
                "source_payload": row,
                "fixture": True,
            },
        )

    @staticmethod
    def _candle_record(asset: str, timestamp: datetime, index: int) -> SourceRecord:
        close = 100.0 + index
        row = {"t": int(timestamp.timestamp() * 1000), "c": str(close)}
        return SourceRecord(
            "MARKET",
            "hyperliquid",
            f"candle:{asset}:1h:{timestamp.isoformat()}",
            f"{asset} 1h candle",
            None,
            [],
            timestamp,
            json.dumps(row),
            {
                "asset": asset,
                "kind": "candle",
                "interval": "1h",
                "candle_close_timestamp": timestamp.isoformat(),
                "close": close,
                "source_payload": row,
                "fixture": True,
            },
        )


def _meta_contexts(payload: Any) -> tuple[list[str], list[dict[str, Any]]]:
    if (
        not isinstance(payload, list)
        or len(payload) < 2
        or not isinstance(payload[0], dict)
    ):
        raise ValueError("Unexpected Hyperliquid metaAndAssetCtxs response")
    universe = [str(item["name"]) for item in payload[0].get("universe", [])]
    contexts = payload[1]
    if not isinstance(contexts, list) or len(contexts) != len(universe):
        raise ValueError("Hyperliquid asset context length mismatch")
    return universe, contexts


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _timestamp(value: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(value) / 1000, UTC)
    except (TypeError, ValueError, OSError):
        return None


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp: {value}") from exc


def record_hash(record: SourceRecord) -> str:
    return hashlib.sha256(
        (record.raw_text + json.dumps(record.raw_metadata, sort_keys=True)).encode()
    ).hexdigest()
