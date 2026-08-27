from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, Field

OutputT = TypeVar("OutputT", bound=BaseModel)


class AnalysisRole(StrEnum):
    TRIAGE = "TRIAGE"
    EXTRACTION = "EXTRACTION"
    HYPOTHESIS_CANDIDATE = "HYPOTHESIS_CANDIDATE"
    TUTOR = "TUTOR"
    ANALYST = "ANALYST"
    CRITIC = "CRITIC"
    WEEKLY_REVIEW = "WEEKLY_REVIEW"


class ModelTier(StrEnum):
    FLASH = "flash"
    PRO = "pro"


@dataclass(frozen=True)
class LLMCallConfig:
    role: AnalysisRole
    tier: ModelTier
    model: str
    thinking: bool
    reasoning_effort: str | None
    max_output_tokens: int
    allow_fallback: bool = False


DEFAULT_ROUTES: dict[AnalysisRole, tuple[ModelTier, bool, str | None, int]] = {
    AnalysisRole.TRIAGE: (ModelTier.FLASH, False, None, 1500),
    AnalysisRole.EXTRACTION: (ModelTier.FLASH, False, None, 2500),
    AnalysisRole.HYPOTHESIS_CANDIDATE: (ModelTier.FLASH, True, "high", 3000),
    AnalysisRole.TUTOR: (ModelTier.FLASH, True, "low", 3000),
    AnalysisRole.ANALYST: (ModelTier.PRO, True, "high", 6000),
    AnalysisRole.CRITIC: (ModelTier.PRO, True, "max", 6000),
    AnalysisRole.WEEKLY_REVIEW: (ModelTier.PRO, True, "max", 10000),
}


class ModelRouter:
    def __init__(
        self,
        flash_model: str = "deepseek-v4-flash",
        pro_model: str = "deepseek-v4-pro",
        routes: dict[AnalysisRole, tuple[ModelTier, bool, str | None, int]]
        | None = None,
    ) -> None:
        if not flash_model or not pro_model:
            raise ValueError("Both flash and pro model configuration are required")
        self.flash_model = flash_model
        self.pro_model = pro_model
        self.routes = DEFAULT_ROUTES if routes is None else routes

    def resolve(self, role: AnalysisRole | str) -> LLMCallConfig:
        try:
            typed_role = role if isinstance(role, AnalysisRole) else AnalysisRole(role)
        except ValueError as exc:
            raise ValueError(f"Unknown analysis role: {role}") from exc
        try:
            tier, thinking, effort, tokens = self.routes[typed_role]
            model = self.flash_model if tier == ModelTier.FLASH else self.pro_model
        except KeyError as exc:
            raise ValueError(
                f"Missing routing configuration for role: {typed_role}"
            ) from exc
        if not model:
            raise ValueError(f"Missing model configuration for tier: {tier}")
        return LLMCallConfig(typed_role, tier, model, thinking, effort, tokens)


class TriageOutput(BaseModel):
    relevance_score: int = Field(ge=0, le=100)
    novelty_score: int = Field(ge=0, le=100)
    testability_score: int = Field(ge=0, le=100)
    executability_score: int = Field(ge=0, le=100)
    latency_sensitivity: str
    reason: str
    retain: bool


class AnalystOutput(BaseModel):
    core_question: str
    reported_finding: str
    mechanism: str
    market: str
    universe: str
    horizon: str
    required_data: list[str]
    possible_hypothesis: str
    practical_reproducibility: str
    unknowns: list[str]


class CriticOutput(BaseModel):
    biases: list[str]
    confounders: list[str]
    failure_reasons: list[str]
    provenance_sufficient: bool


class TutorConcept(BaseModel):
    name: str
    beginner_explanation: str
    technical_definition: str
    why_it_matters: str
    example: str
    question: str


class TutorOutput(BaseModel):
    concepts: list[TutorConcept] = Field(max_length=2)


class LLMClient(Protocol):
    provider: str
    model: str
    prompt_version: str

    def triage(self, title: str, text: str) -> TriageOutput: ...
    def analyze(self, title: str, text: str) -> AnalystOutput: ...
    def critique(self, hypothesis: str) -> CriticOutput: ...
    def tutor(self, hypothesis: str) -> TutorOutput: ...


class DeepSeekClient:
    provider = "deepseek"
    prompt_version = "phase15c-1"

    def __init__(
        self,
        api_key: str,
        router: ModelRouter | str,
        base_url: str = "https://api.deepseek.com",
        timeout: float = 30.0,
        retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self.router = (
            router if isinstance(router, ModelRouter) else ModelRouter(router, router)
        )
        self.model = self.router.flash_model
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout
        self._retries = retries
        self._client = client or httpx.Client(timeout=timeout)

    def call(
        self, role: AnalysisRole, schema: type[OutputT], instruction: str, text: str
    ) -> OutputT:
        config = self.router.resolve(role)
        body: dict[str, Any] = {
            "model": config.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": f"{instruction} Return only valid JSON matching this schema: {schema.model_json_schema()}",
                },
                {"role": "user", "content": text},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": config.max_output_tokens,
        }
        if config.thinking:
            body["thinking"] = {"type": "enabled"}
            body["reasoning_effort"] = config.reasoning_effort
        else:
            body["thinking"] = {"type": "disabled"}
        last: Exception | None = None
        for _ in range(self._retries + 1):
            try:
                response = self._client.post(
                    self._url, headers=self._headers, json=body, timeout=self._timeout
                )
                response.raise_for_status()
                message = response.json()["choices"][0]["message"]
                content = message["content"]
                return schema.model_validate(json.loads(content))
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                last = exc
        raise ValueError(
            f"LLM structured output failed validation or request: {last}"
        ) from last

    def triage(self, title: str, text: str) -> TriageOutput:
        return self.call(
            AnalysisRole.TRIAGE,
            TriageOutput,
            "Assess source relevance, novelty, testability, and execution feasibility.",
            f"Title: {title}\n{text}",
        )

    def analyze(self, title: str, text: str) -> AnalystOutput:
        return self.call(
            AnalysisRole.ANALYST,
            AnalystOutput,
            "Separate source-reported findings from interpretation and preserve uncertainty.",
            f"Title: {title}\n{text}",
        )

    def critique(self, hypothesis: str) -> CriticOutput:
        return self.call(
            AnalysisRole.CRITIC,
            CriticOutput,
            "Adversarially assess bias, confounding, PIT, and provenance limitations.",
            hypothesis,
        )

    def tutor(self, hypothesis: str) -> TutorOutput:
        return self.call(
            AnalysisRole.TUTOR,
            TutorOutput,
            "Explain the quantitative concepts for a beginner.",
            hypothesis,
        )


OpenAICompatClient = DeepSeekClient


class FakeLLMClient:
    provider = "fake"
    model = "fake-v1"
    prompt_version = "fixture-1"

    def triage(self, title: str, text: str) -> TriageOutput:
        return TriageOutput(
            relevance_score=75,
            novelty_score=55,
            testability_score=80,
            executability_score=70,
            latency_sensitivity="UNKNOWN",
            reason="Fixture analysis: source describes a measurable market relationship.",
            retain=True,
        )

    def analyze(self, title: str, text: str) -> AnalystOutput:
        return AnalystOutput(
            core_question=title,
            reported_finding=text,
            mechanism="Funding pressure may reflect positioning and liquidity imbalance.",
            market="Crypto perpetuals",
            universe="BTC, ETH, SOL",
            horizon="1h to 24h",
            required_data=["funding rate", "mark price", "point-in-time timestamps"],
            possible_hypothesis="Extreme funding observations are associated with different subsequent returns than ordinary funding observations.",
            practical_reproducibility="Potentially reproducible with timestamped public data.",
            unknowns=["Costs and sample stability are unknown."],
        )

    def critique(self, hypothesis: str) -> CriticOutput:
        return CriticOutput(
            biases=[
                "look-ahead bias",
                "data snooping",
                "regime dependence",
                "transaction costs",
            ],
            confounders=["market beta", "liquidity conditions"],
            failure_reasons=["Historical point-in-time coverage must be verified."],
            provenance_sufficient=True,
        )

    def tutor(self, hypothesis: str) -> TutorOutput:
        return TutorOutput(
            concepts=[
                TutorConcept(
                    name="Funding rate",
                    beginner_explanation="A periodic payment between perpetual-futures traders that helps keep the contract near its reference price.",
                    technical_definition="The signed periodic funding payment rate applied to notional exposure.",
                    why_it_matters="The hypothesis uses unusually high or low funding as its independent variable.",
                    example="A rate of 0.01% on $10,000 notional is $1 for that funding interval.",
                    question="Why must the funding timestamp be recorded before measuring a later return?",
                )
            ]
        )
