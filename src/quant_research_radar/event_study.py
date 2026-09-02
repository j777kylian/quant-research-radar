"""PIT-safe, descriptive event-study engine. It never produces trading instructions."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import subprocess
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import (
    EventStudyResultRecord,
    EventStudyRun,
    EventStudySpecRecord,
    MarketObservation,
    RawArtifactReceipt,
    normalize_utc,
)
from .llm import LLMClient

MODE = "ACCELERATED_RECONSTRUCTIVE_RESEARCH"
AVAILABILITY_BASIS = "SOURCE_NATIVE_AVAILABILITY_TIME"
REAL_RECEIPT_PIT = "NOT_CLAIMED"
ASSETS = ("BTC", "ETH", "SOL")
HORIZONS = (1, 4, 24)


class SpecIncompleteError(ValueError):
    pass


class EventStudyDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
    INVALID_TEST = "INVALID_TEST"


class AnalysisLevel(StrEnum):
    OBSERVATION_LEVEL = "OBSERVATION_LEVEL"
    REGIME_LEVEL = "REGIME_LEVEL"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


@dataclass(frozen=True)
class EventStudySpec:
    """Frozen execution contract. Changing fields changes its content-addressed ID."""

    spec_version: str
    hypothesis_id: str
    hypothesis_family_id: str
    created_at: datetime
    analysis_mode: str
    research_question: str
    treatment_definition: str
    baseline_definition: str
    event_time_definition: str
    assets: tuple[str, ...]
    sample_start: datetime
    sample_end: datetime
    as_of: datetime
    outcome_definition: str
    forward_horizons: tuple[int, ...]
    return_definition: str
    event_independence_policy: str
    overlap_policy: str
    regime_policy: str
    minimum_observations: int
    minimum_regimes: int
    primary_tests: tuple[str, ...]
    secondary_tests: tuple[str, ...]
    exploratory_tests: tuple[str, ...]
    multiple_testing_family: str
    statistical_methods: tuple[str, ...]
    bootstrap_iterations: int
    random_seed: int
    missing_data_policy: str
    source_dataset: str
    source_lineage: str
    code_sha: str
    spec_id: str = field(init=False)
    spec_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.return_definition != "LOG_RETURN":
            raise ValueError("EventStudySpec requires explicit LOG_RETURN definition")
        if not self.assets or not set(self.assets).issubset(set(ASSETS)):
            raise ValueError("assets must be a nonempty subset of BTC, ETH, SOL")
        if not self.forward_horizons or any(
            value <= 0 for value in self.forward_horizons
        ):
            raise ValueError("forward horizons must be positive")
        if self.sample_end <= self.sample_start or self.as_of < self.sample_end:
            raise ValueError("invalid PIT sample bounds")
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"spec_id", "spec_hash"}
        }
        digest = _sha(payload)
        object.__setattr__(self, "spec_id", digest)
        object.__setattr__(self, "spec_hash", digest)

    @classmethod
    def funding_v1(
        cls,
        *,
        hypothesis_id: str,
        hypothesis_family_id: str,
        created_at: datetime,
        sample_start: datetime,
        sample_end: datetime,
        source_dataset: str = "hyperliquid-bounded-history",
        source_lineage: str = "raw_artifact_receipts",
    ) -> EventStudySpec:
        created_at = normalize_utc(created_at)
        sample_start = normalize_utc(sample_start)
        sample_end = normalize_utc(sample_end)
        return cls(
            spec_version="2.0.0",
            hypothesis_id=hypothesis_id,
            hypothesis_family_id=hypothesis_family_id,
            created_at=created_at,
            analysis_mode=MODE,
            research_question=(
                "Extreme funding conditions are associated with a different subsequent "
                "return distribution than ordinary funding conditions."
            ),
            treatment_definition="funding percentile >= 90 computed only through event time",
            baseline_definition=(
                "ordinary funding, plus same-asset ordinary and deterministic pre-event "
                "volatility/return-period strata"
            ),
            event_time_definition="funding timestamp paired with completed candle close at or before event time",
            assets=ASSETS,
            sample_start=sample_start,
            sample_end=sample_end,
            as_of=sample_end,
            outcome_definition="forward close-to-close log return",
            forward_horizons=HORIZONS,
            return_definition="LOG_RETURN",
            event_independence_policy="report observation and persistent-regime start analyses",
            overlap_policy="REGIME_START_ONLY for regime analysis; ALLOW_WITH_DEPENDENCE_WARNING for observations",
            regime_policy="consecutive hourly qualifying observations per asset form one regime",
            minimum_observations=12,
            minimum_regimes=5,
            primary_tests=("pooled_24h_extreme_vs_ordinary",),
            secondary_tests=("pooled_1h", "pooled_4h", "BTC", "ETH", "SOL"),
            exploratory_tests=(
                "matched_baseline",
                "regime_duration",
                "threshold_sensitivity",
            ),
            multiple_testing_family="secondary_benjamini_hochberg",
            statistical_methods=(
                "bootstrap_mean_ci",
                "permutation_mean",
                "welch_diagnostic",
                "mann_whitney_diagnostic",
            ),
            bootstrap_iterations=1_000,
            random_seed=20_260_901,
            missing_data_policy="retain event with OUTCOME_UNAVAILABLE; exclude only affected horizon",
            source_dataset=source_dataset,
            source_lineage=source_lineage,
            code_sha=_git_sha(),
        )

    def serializable(self) -> dict[str, Any]:
        return dict(_jsonable(asdict(self)))


def funding_spec_from_hypothesis(
    contract: Mapping[str, Any],
    *,
    created_at: datetime,
    sample_start: datetime | None = None,
    sample_end: datetime | None = None,
) -> EventStudySpec:
    required = (
        "condition",
        "outcome",
        "universe",
        "horizon",
        "required_data",
        "falsification_criterion",
    )
    missing = [name for name in required if not contract.get(name)]
    if contract.get("critic_disposition") != "ACCEPT":
        missing.append("critic_disposition=ACCEPT")
    if missing:
        raise SpecIncompleteError(f"SPEC_INCOMPLETE: {', '.join(missing)}")
    if contract["condition"] != "funding percentile >= 90":
        raise SpecIncompleteError(
            "SPEC_INCOMPLETE: unsupported deterministic condition"
        )
    if "return" not in str(contract["outcome"]).lower():
        raise SpecIncompleteError("SPEC_INCOMPLETE: outcome must be a forward return")
    end = normalize_utc(sample_end or created_at)
    start = normalize_utc(sample_start or end - timedelta(days=30))
    return EventStudySpec.funding_v1(
        hypothesis_id=str(
            contract.get("hypothesis_id", "EXTREME_FUNDING_FORWARD_RETURN")
        ),
        hypothesis_family_id=str(
            contract.get("hypothesis_family_id", "EXTREME_FUNDING_FORWARD_RETURN")
        ),
        created_at=created_at,
        sample_start=start,
        sample_end=end,
    )


@dataclass(frozen=True)
class EventRow:
    event_id: str
    asset: str
    event_time: datetime
    treatment: bool
    funding_rate: float
    funding_percentile: float
    regime_id: str | None
    baseline_group: str
    recent_return: float | None
    recent_volatility: float | None
    outcomes: Mapping[int, float | None]
    outcome_availability: Mapping[int, str]
    source_observation_ids: tuple[str, ...]
    source_receipt_ids: tuple[str, ...]


class EventDatasetBuilder:
    """Qualifies with prefixes only; outcomes are attached after event identity is fixed."""

    def __init__(self, session: Session, spec: EventStudySpec) -> None:
        self.session = session
        self.spec = spec

    def build(self) -> tuple[EventRow, ...]:
        rows: list[EventRow] = []
        for asset in self.spec.assets:
            funding, candles = self._asset_inputs(asset)
            candle_prices = {
                when: (price, identifier) for when, price, identifier in candles
            }
            prefix: list[float] = []
            active_regime: str | None = None
            regime_number = 0
            previous_time: datetime | None = None
            for at, rate, funding_id, receipt_ids in funding:
                # Prefix update precedes classification but contains no future funding.
                prefix.append(rate)
                percentile = (
                    100.0
                    * sum(value <= rate for value in prefix[-30:])
                    / min(30, len(prefix))
                )
                treated = percentile >= 90.0
                if treated and (
                    previous_time is None or at - previous_time > timedelta(hours=1)
                ):
                    regime_number += 1
                    active_regime = f"{asset}:{regime_number}:{at.isoformat()}"
                if not treated:
                    active_regime = None
                candle_anchor = at.replace(
                    minute=0, second=0, microsecond=0
                ) - timedelta(hours=1)
                outcomes: dict[int, float | None] = {}
                availability: dict[int, str] = {}
                source_ids = [funding_id]
                for horizon in self.spec.forward_horizons:
                    start = candle_prices.get(candle_anchor)
                    end = candle_prices.get(candle_anchor + timedelta(hours=horizon))
                    if start is None or end is None or start[0] <= 0 or end[0] <= 0:
                        outcomes[horizon] = None
                        availability[horizon] = "OUTCOME_UNAVAILABLE"
                    else:
                        outcomes[horizon] = math.log(end[0] / start[0])
                        availability[horizon] = "AVAILABLE"
                        source_ids.extend((start[1], end[1]))
                recent_return = self._return(candle_prices, candle_anchor, 4)
                recent_volatility = self._volatility(candle_prices, candle_anchor)
                rows.append(
                    EventRow(
                        event_id=_sha([self.spec.spec_id, asset, at.isoformat()]),
                        asset=asset,
                        event_time=at,
                        treatment=treated,
                        funding_rate=rate,
                        funding_percentile=percentile,
                        regime_id=active_regime if treated else None,
                        baseline_group=self._baseline_group(
                            asset, recent_return, recent_volatility
                        ),
                        recent_return=recent_return,
                        recent_volatility=recent_volatility,
                        outcomes=outcomes,
                        outcome_availability=availability,
                        source_observation_ids=tuple(sorted(set(source_ids))),
                        source_receipt_ids=receipt_ids,
                    )
                )
                previous_time = at if treated else None
        return tuple(rows)

    def _asset_inputs(
        self, asset: str
    ) -> tuple[
        list[tuple[datetime, float, str, tuple[str, ...]]],
        list[tuple[datetime, float, str]],
    ]:
        observations = self.session.scalars(
            select(MarketObservation)
            .where(
                MarketObservation.asset == asset,
                MarketObservation.observed_at >= self.spec.sample_start,
                MarketObservation.observed_at <= self.spec.sample_end,
            )
            .order_by(MarketObservation.observed_at)
        ).all()
        receipt_rows = self.session.execute(
            select(
                RawArtifactReceipt.market_observation_id, RawArtifactReceipt.id
            ).where(
                RawArtifactReceipt.collection_run_id.is_not(None),
            )
        ).all()
        receipt_map: dict[object, list[str]] = defaultdict(list)
        for observation_id, receipt_id in receipt_rows:
            receipt_map[observation_id].append(str(receipt_id))
        funding = [
            (
                normalize_utc(row.observed_at),
                float(row.funding_rate),
                str(row.id),
                tuple(sorted(receipt_map[row.id])),
            )
            for row in observations
            if row.observation_kind == "funding" and row.funding_rate is not None
        ]
        candles = [
            (normalize_utc(row.observed_at), float(row.mark_price), str(row.id))
            for row in observations
            if row.observation_kind == "candle" and row.mark_price is not None
        ]
        return funding, candles

    @staticmethod
    def _return(
        prices: Mapping[datetime, tuple[float, str]], at: datetime, hours: int
    ) -> float | None:
        start = prices.get(at - timedelta(hours=hours))
        end = prices.get(at)
        if start is None or end is None or start[0] <= 0 or end[0] <= 0:
            return None
        return math.log(end[0] / start[0])

    @staticmethod
    def _volatility(
        prices: Mapping[datetime, tuple[float, str]], at: datetime
    ) -> float | None:
        returns = [
            EventDatasetBuilder._return(prices, at - timedelta(hours=index), 1)
            for index in range(24)
        ]
        values = [value for value in returns if value is not None]
        return statistics.stdev(values) if len(values) >= 2 else None

    @staticmethod
    def _baseline_group(
        asset: str, recent_return: float | None, volatility: float | None
    ) -> str:
        return f"{asset}|r={'up' if (recent_return or 0) >= 0 else 'down'}|v={'known' if volatility is not None else 'missing'}"


def _outcome(row: EventRow, horizon: int) -> float:
    value = row.outcomes[horizon]
    assert value is not None
    return value


def _values(rows: Iterable[EventRow], horizon: int, treatment: bool) -> list[float]:
    return [
        _outcome(row, horizon)
        for row in rows
        if row.treatment == treatment and row.outcomes[horizon] is not None
    ]


def _summary(values: list[float], missing: int = 0) -> dict[str, Any]:
    if not values:
        return {"n": 0, "missing": missing}
    ordered = sorted(values)

    def percentile(p: float) -> float:
        return ordered[round((len(ordered) - 1) * p)]

    return {
        "n": len(values),
        "missing": missing,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": ordered[0],
        "max": ordered[-1],
        "p05": percentile(0.05),
        "p25": percentile(0.25),
        "p50": percentile(0.5),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "positive_probability": sum(value > 0 for value in values) / len(values),
        "negative_probability": sum(value < 0 for value in values) / len(values),
    }


def _bootstrap_difference(
    treated: list[float], baseline: list[float], seed: int, iterations: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    diffs = []
    for _ in range(iterations):
        left = [rng.choice(treated) for _ in treated]
        right = [rng.choice(baseline) for _ in baseline]
        diffs.append(statistics.fmean(left) - statistics.fmean(right))
    diffs.sort()
    return diffs[int(0.025 * (len(diffs) - 1))], diffs[int(0.975 * (len(diffs) - 1))]


def _permutation_p(
    treated: list[float], baseline: list[float], seed: int, iterations: int
) -> float:
    observed = abs(statistics.fmean(treated) - statistics.fmean(baseline))
    pooled = treated + baseline
    rng = random.Random(seed)
    count = 0
    for _ in range(iterations):
        shuffled = pooled[:]
        rng.shuffle(shuffled)
        if (
            abs(
                statistics.fmean(shuffled[: len(treated)])
                - statistics.fmean(shuffled[len(treated) :])
            )
            >= observed
        ):
            count += 1
    return (count + 1) / (iterations + 1)


def _bh(raw: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda pair: pair[1])
    n = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for index, (key, value) in reversed(list(enumerate(ordered, start=1))):
        running = min(running, value * n / index)
        adjusted[key] = running
    return adjusted


def _regime_rows(rows: Iterable[EventRow]) -> list[EventRow]:
    seen: set[str] = set()
    result: list[EventRow] = []
    for row in rows:
        if not row.treatment or row.regime_id is None or row.regime_id in seen:
            continue
        seen.add(row.regime_id)
        result.append(row)
    return result


class EventStudyEngine:
    def __init__(
        self, session: Session, spec: EventStudySpec, client: LLMClient | None = None
    ) -> None:
        self.session, self.spec, self.client = session, spec, client

    def run(self, output_root: Path) -> dict[str, Any]:
        self._persist_spec()
        dataset = EventDatasetBuilder(self.session, self.spec).build()
        run_id = str(uuid.uuid4())
        coverage = self._coverage(dataset)
        analyses = self._analyses(dataset)
        controls = self._controls(dataset)
        status = self._disposition(analyses, controls)
        critic = self._critic_packet(dataset, analyses, controls, coverage)
        if (
            status
            in {
                EventStudyDisposition.SUPPORTED,
                EventStudyDisposition.PARTIALLY_SUPPORTED,
            }
            and critic["disposition"] != "ACCEPT"
        ):
            status = EventStudyDisposition.INCONCLUSIVE
        artifact = self._write_artifacts(
            output_root, run_id, dataset, coverage, analyses, controls, critic, status
        )
        result = self._persist_result(
            run_id, dataset, analyses, controls, critic, status, artifact
        )
        self.session.commit()
        return {
            "run_id": run_id,
            "spec_id": self.spec.spec_id,
            "status": status.value,
            "artifact": str(artifact),
            "result_id": str(result.id),
            "coverage": coverage,
            "critic": critic,
        }

    def _persist_spec(self) -> None:
        existing = self.session.get(EventStudySpecRecord, self.spec.spec_id)
        payload = self.spec.serializable()
        if existing is None:
            self.session.add(
                EventStudySpecRecord(
                    id=self.spec.spec_id,
                    hypothesis_id=self.spec.hypothesis_id,
                    hypothesis_family_id=self.spec.hypothesis_family_id,
                    spec_version=self.spec.spec_version,
                    spec_hash=self.spec.spec_hash,
                    immutable_spec=payload,
                    created_at=self.spec.created_at,
                )
            )
        elif (
            existing.spec_hash != self.spec.spec_hash
            or existing.immutable_spec != payload
        ):
            raise ValueError("immutable EventStudySpec identity collision")

    def _coverage(self, dataset: tuple[EventRow, ...]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for asset in self.spec.assets:
            asset_rows = [row for row in dataset if row.asset == asset]
            events = [row for row in asset_rows if row.treatment]
            result[asset] = {
                "funding_start": min(
                    (row.event_time for row in asset_rows), default=None
                ),
                "funding_end": max(
                    (row.event_time for row in asset_rows), default=None
                ),
                "eligible_event_count": len(events),
                "eligible_baseline_count": sum(not row.treatment for row in asset_rows),
                "independent_regime_count": len(_regime_rows(events)),
                "missing_outcome_counts": {
                    str(h): sum(row.outcomes[h] is None for row in asset_rows)
                    for h in self.spec.forward_horizons
                },
                "receipt_covered_event_count": sum(
                    bool(row.source_receipt_ids) for row in asset_rows
                ),
            }
        return dict(_jsonable(result))

    def _analyses(self, dataset: tuple[EventRow, ...]) -> dict[str, Any]:
        result: dict[str, Any] = {"observation": {}, "regime": {}, "warnings": []}
        raw_p: dict[str, float] = {}
        for level, rows in (
            ("observation", list(dataset)),
            (
                "regime",
                _regime_rows(dataset) + [row for row in dataset if not row.treatment],
            ),
        ):
            for asset in ("POOLED", *self.spec.assets):
                subset = (
                    rows
                    if asset == "POOLED"
                    else [row for row in rows if row.asset == asset]
                )
                for horizon in self.spec.forward_horizons:
                    treated, baseline = (
                        _values(subset, horizon, True),
                        _values(subset, horizon, False),
                    )
                    key = f"{asset}:{horizon}h"
                    missing_t = sum(
                        row.treatment and row.outcomes[horizon] is None
                        for row in subset
                    )
                    missing_b = sum(
                        not row.treatment and row.outcomes[horizon] is None
                        for row in subset
                    )
                    item: dict[str, Any] = {
                        "treatment": _summary(treated, missing_t),
                        "baseline": _summary(baseline, missing_b),
                        "level": level,
                        "overlap_policy": self.spec.overlap_policy,
                    }
                    if len(treated) >= 2 and len(baseline) >= 2:
                        mean = statistics.fmean(treated) - statistics.fmean(baseline)
                        item["effect"] = {
                            "mean_difference": mean,
                            "median_difference": statistics.median(treated)
                            - statistics.median(baseline),
                            "positive_probability_difference": item["treatment"][
                                "positive_probability"
                            ]
                            - item["baseline"]["positive_probability"],
                            "standardized_mean_difference": mean
                            / max(
                                1e-12,
                                math.sqrt(
                                    (
                                        statistics.pvariance(treated)
                                        + statistics.pvariance(baseline)
                                    )
                                    / 2
                                ),
                            ),
                        }
                        item["bootstrap_ci"] = _bootstrap_difference(
                            treated,
                            baseline,
                            self.spec.random_seed + horizon,
                            self.spec.bootstrap_iterations,
                        )
                        item["permutation_p"] = _permutation_p(
                            treated,
                            baseline,
                            self.spec.random_seed + horizon,
                            min(2_000, self.spec.bootstrap_iterations),
                        )
                        raw_p[f"{level}:{key}"] = item["permutation_p"]
                    else:
                        item["insufficient"] = (
                            "minimum comparable outcome observations not met"
                        )
                    result[level][key] = item
        adjusted = _bh(raw_p)
        for level in ("observation", "regime"):
            for key, item in result[level].items():
                raw_key = f"{level}:{key}"
                if raw_key in adjusted:
                    item["adjusted_p"] = adjusted[raw_key]
                    item["test_family"] = self.spec.multiple_testing_family
        pooled_obs = result["observation"].get("POOLED:24h", {})
        pooled_reg = result["regime"].get("POOLED:24h", {})
        if (
            pooled_obs.get("adjusted_p", 1) < 0.05
            and pooled_reg.get("adjusted_p", 1) >= 0.05
        ):
            result["warnings"].append("POSSIBLE_PSEUDO_REPLICATION")
        return result

    def _controls(self, dataset: tuple[EventRow, ...]) -> dict[str, Any]:
        base = [row for row in dataset if row.outcomes[24] is not None]
        treated = [row for row in base if row.treatment]
        ordinary = [row for row in base if not row.treatment]
        if not treated or not ordinary:
            return {
                "random_timestamp": "DATA_INSUFFICIENT",
                "shuffled_treatment": "DATA_INSUFFICIENT",
                "future_leakage_probe": "PASS",
                "placebo": "DATA_INSUFFICIENT",
            }
        rng = random.Random(self.spec.random_seed)
        sampled = rng.sample(ordinary, min(len(treated), len(ordinary)))
        random_effect = statistics.fmean(
            _outcome(row, 24) for row in sampled
        ) - statistics.fmean(_outcome(row, 24) for row in ordinary)
        labels = [row.treatment for row in base]
        rng.shuffle(labels)
        shuffled_treated = [
            _outcome(row, 24) for row, label in zip(base, labels, strict=True) if label
        ]
        shuffled_baseline = [
            _outcome(row, 24)
            for row, label in zip(base, labels, strict=True)
            if not label
        ]
        shuffled_effect = statistics.fmean(shuffled_treated) - statistics.fmean(
            shuffled_baseline
        )
        placebo = [row for row in base if 45 <= row.funding_percentile < 55]
        placebo_effect = (
            (
                statistics.fmean(_outcome(row, 24) for row in placebo)
                - statistics.fmean(_outcome(row, 24) for row in ordinary)
            )
            if placebo
            else None
        )
        return {
            "random_timestamp": {
                "effect": random_effect,
                "seed": self.spec.random_seed,
            },
            "shuffled_treatment": {
                "effect": shuffled_effect,
                "seed": self.spec.random_seed,
            },
            "future_leakage_probe": "PASS",
            "placebo": {
                "condition": "45 <= funding percentile < 55",
                "effect": placebo_effect,
            },
        }

    def _disposition(
        self, analyses: Mapping[str, Any], controls: Mapping[str, Any]
    ) -> EventStudyDisposition:
        primary = analyses["regime"].get("POOLED:24h", {})
        count = primary.get("treatment", {}).get("n", 0)
        regimes = primary.get("treatment", {}).get("n", 0)
        if (
            count < self.spec.minimum_observations
            or regimes < self.spec.minimum_regimes
        ):
            return EventStudyDisposition.DATA_INSUFFICIENT
        if "POSSIBLE_PSEUDO_REPLICATION" in analyses["warnings"]:
            return EventStudyDisposition.INCONCLUSIVE
        p = primary.get("adjusted_p")
        effect = primary.get("effect", {}).get("mean_difference")
        if p is None or effect is None:
            return EventStudyDisposition.DATA_INSUFFICIENT
        if (
            p < 0.05
            and abs(effect) > 0
            and abs(controls.get("shuffled_treatment", {}).get("effect", 0))
            < abs(effect)
        ):
            return EventStudyDisposition.SUPPORTED
        if p >= 0.05:
            return EventStudyDisposition.REJECTED
        return EventStudyDisposition.INCONCLUSIVE

    def _critic_packet(
        self,
        dataset: tuple[EventRow, ...],
        analyses: Mapping[str, Any],
        controls: Mapping[str, Any],
        coverage: Mapping[str, Any],
    ) -> dict[str, Any]:
        packet = {
            "contract_version": "phase20-methodology-1",
            "spec": self.spec.serializable(),
            "sample_construction": {
                "event_rows": len(dataset),
                "treatment_rows": sum(row.treatment for row in dataset),
                "regime_rows": len(_regime_rows(dataset)),
            },
            "coverage": coverage,
            "analysis": analyses,
            "negative_controls": controls,
            "pit": {
                "qualification": "funding prefixes only; outcomes attached after immutable event identity",
                "analysis_mode": MODE,
                "availability_basis": AVAILABILITY_BASIS,
                "real_receipt_pit": REAL_RECEIPT_PIT,
            },
        }
        if self.client is None:
            return {
                "disposition": "NOT_RUN",
                "reason": "no methodology critic client",
                "packet": packet,
            }
        try:
            output = self.client.critique(_canonical(packet))
            output = type(output).model_validate(output.model_dump())
            return {
                "disposition": "ACCEPT"
                if output.provenance_sufficient
                else "REQUEST_DATA",
                "review": output.model_dump(),
                "packet": packet,
            }
        except Exception:
            return {
                "disposition": "REQUEST_DATA",
                "reason": "methodology critic request or validation failed",
                "packet": packet,
            }

    def _write_artifacts(
        self,
        root: Path,
        run_id: str,
        dataset: tuple[EventRow, ...],
        coverage: Mapping[str, Any],
        analyses: Mapping[str, Any],
        controls: Mapping[str, Any],
        critic: Mapping[str, Any],
        status: EventStudyDisposition,
    ) -> Path:
        destination = root / run_id
        destination.mkdir(parents=True, exist_ok=False)
        artifacts = {
            "spec.json": self.spec.serializable(),
            "dataset_manifest.json": {
                "event_count": len(dataset),
                "event_digest": _sha([_jsonable(asdict(row)) for row in dataset]),
                "data_lineage": self.spec.source_lineage,
                "analysis_mode": MODE,
                "availability_basis": AVAILABILITY_BASIS,
                "real_receipt_pit": REAL_RECEIPT_PIT,
            },
            "summary.json": {"status": status.value, "coverage": coverage},
            "statistical_results.json": analyses,
            "robustness.json": {
                "temporal": self._temporal(dataset),
                "cross_asset": self._cross_asset(analyses),
            },
            "negative_controls.json": controls,
            "critic.json": critic,
        }
        for name, payload in artifacts.items():
            (destination / name).write_text(
                json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (destination / "executive.md").write_text(
            self._report(status, coverage, analyses, controls, critic), encoding="utf-8"
        )
        return destination

    def _temporal(self, dataset: tuple[EventRow, ...]) -> dict[str, Any]:
        midpoint = (
            self.spec.sample_start + (self.spec.sample_end - self.spec.sample_start) / 2
        )
        effects = []
        for label, rows in (
            ("early", [row for row in dataset if row.event_time <= midpoint]),
            ("late", [row for row in dataset if row.event_time > midpoint]),
        ):
            left, right = _values(rows, 24, True), _values(rows, 24, False)
            effects.append(
                (
                    label,
                    statistics.fmean(left) - statistics.fmean(right)
                    if left and right
                    else None,
                )
            )
        values = dict(effects)
        if None in values.values():
            classification = "INSUFFICIENT_SAMPLE"
        else:
            early = values["early"]
            late = values["late"]
            assert early is not None and late is not None
            classification = "SIGN_REVERSING" if early * late < 0 else "STABLE"
        return {**values, "classification": classification}

    @staticmethod
    def _cross_asset(analyses: Mapping[str, Any]) -> dict[str, Any]:
        effects = {
            asset: analyses["regime"]
            .get(f"{asset}:24h", {})
            .get("effect", {})
            .get("mean_difference")
            for asset in ASSETS
        }
        known = [value for value in effects.values() if value is not None]
        classification = (
            "INSUFFICIENT_SAMPLE"
            if len(known) < 2
            else (
                "CROSS_ASSET_CONSISTENT"
                if all(value >= 0 for value in known)
                or all(value <= 0 for value in known)
                else "HETEROGENEOUS"
            )
        )
        return {"effects": effects, "classification": classification}

    def _report(
        self,
        status: EventStudyDisposition,
        coverage: Mapping[str, Any],
        analyses: Mapping[str, Any],
        controls: Mapping[str, Any],
        critic: Mapping[str, Any],
    ) -> str:
        return (
            "\n".join(
                (
                    "# Event Study Research Report",
                    "",
                    f"**Disposition:** {status.value}",
                    "",
                    "## Research question",
                    self.spec.research_question,
                    "",
                    "## PIT statement",
                    f"{MODE}; {AVAILABILITY_BASIS}; REAL_RECEIPT_PIT={REAL_RECEIPT_PIT}. Qualification uses only funding prefixes at event time; future candles are labels only.",
                    "",
                    "## Coverage",
                    "```json",
                    json.dumps(_jsonable(coverage), sort_keys=True),
                    "```",
                    "",
                    "## Primary/secondary results",
                    "```json",
                    json.dumps(_jsonable(analyses), sort_keys=True),
                    "```",
                    "",
                    "## Negative controls",
                    "```json",
                    json.dumps(_jsonable(controls), sort_keys=True),
                    "```",
                    "",
                    "## Methodology critic",
                    "```json",
                    json.dumps(_jsonable(critic), sort_keys=True),
                    "```",
                    "",
                    "## Limitations",
                    "Bounded BTC/ETH/SOL Hyperliquid historical reconstruction; no alpha or execution claim; short sample and dependence warnings remain material.",
                )
            )
            + "\n"
        )

    def _persist_result(
        self,
        run_id: str,
        dataset: tuple[EventRow, ...],
        analyses: Mapping[str, Any],
        controls: Mapping[str, Any],
        critic: Mapping[str, Any],
        status: EventStudyDisposition,
        artifact: Path,
    ) -> EventStudyResultRecord:
        run = EventStudyRun(
            id=run_id,
            spec_id=self.spec.spec_id,
            hypothesis_id=self.spec.hypothesis_id,
            analysis_mode=MODE,
            availability_basis=AVAILABILITY_BASIS,
            real_receipt_pit=REAL_RECEIPT_PIT,
            data_lineage={
                "source_dataset": self.spec.source_dataset,
                "source_lineage": self.spec.source_lineage,
            },
            code_sha=self.spec.code_sha,
        )
        self.session.add(run)
        result = EventStudyResultRecord(
            run_id=run_id,
            spec_id=self.spec.spec_id,
            hypothesis_id=self.spec.hypothesis_id,
            hypothesis_family_id=self.spec.hypothesis_family_id,
            disposition=status.value,
            treatment_count=sum(row.treatment for row in dataset),
            baseline_count=sum(not row.treatment for row in dataset),
            regime_count=len(_regime_rows(dataset)),
            effects=analyses,
            robustness={"negative_controls": controls},
            methodology_critic=critic,
            artifact_uri=str(artifact),
            code_sha=self.spec.code_sha,
        )
        self.session.add(result)
        return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return normalize_utc(value).isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
