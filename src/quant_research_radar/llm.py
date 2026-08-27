from __future__ import annotations

import json
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, Field

OutputT = TypeVar("OutputT", bound=BaseModel)


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


class OpenAICompatClient:
    provider = "openai-compatible"
    prompt_version = "1"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        retries: int = 2,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout
        self._retries = retries
        self._client = client or httpx.Client(timeout=timeout)

    def _call(self, schema: type[OutputT], instruction: str, text: str) -> OutputT:
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
            "response_format": {"type": "json_object"},
        }
        last: Exception | None = None
        for _ in range(self._retries + 1):
            try:
                response = self._client.post(
                    self._url, headers=self._headers, json=body, timeout=self._timeout
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
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
        return self._call(
            TriageOutput,
            "Return only valid JSON matching the supplied schema.",
            f"Title: {title}\n{text}",
        )

    def analyze(self, title: str, text: str) -> AnalystOutput:
        return self._call(
            AnalystOutput,
            "Return only valid JSON. Preserve reported findings as claims, not facts.",
            f"Title: {title}\n{text}",
        )

    def critique(self, hypothesis: str) -> CriticOutput:
        return self._call(
            CriticOutput,
            "Return only valid JSON assessing limitations and provenance.",
            hypothesis,
        )

    def tutor(self, hypothesis: str) -> TutorOutput:
        return self._call(
            TutorOutput, "Return only valid JSON with educational concepts.", hypothesis
        )


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
