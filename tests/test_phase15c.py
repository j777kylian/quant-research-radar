import json

import httpx
import pytest

from quant_research_radar.llm import (
    AnalysisRole,
    DeepSeekClient,
    ModelRouter,
    TriageOutput,
)


def test_default_role_routes() -> None:
    router = ModelRouter()
    expected = {
        AnalysisRole.TRIAGE: ("deepseek-v4-flash", False, None),
        AnalysisRole.EXTRACTION: ("deepseek-v4-flash", False, None),
        AnalysisRole.HYPOTHESIS_CANDIDATE: ("deepseek-v4-flash", True, "high"),
        AnalysisRole.TUTOR: ("deepseek-v4-flash", True, "low"),
        AnalysisRole.ANALYST: ("deepseek-v4-pro", True, "high"),
        AnalysisRole.CRITIC: ("deepseek-v4-pro", True, "max"),
        AnalysisRole.WEEKLY_REVIEW: ("deepseek-v4-pro", True, "max"),
    }
    for role, values in expected.items():
        config = router.resolve(role)
        assert (config.model, config.thinking, config.reasoning_effort) == values


def test_unknown_and_missing_routes_fail_closed() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        ModelRouter().resolve("unknown")
    with pytest.raises(ValueError, match="Missing routing"):
        ModelRouter(routes={}).resolve(AnalysisRole.TRIAGE)


def test_mock_request_contains_routed_configuration() -> None:
    captured: dict[str, object] = {}
    content = {
        "relevance_score": 1,
        "novelty_score": 2,
        "testability_score": 3,
        "executability_score": 4,
        "latency_sensitivity": "UNKNOWN",
        "reason": "ok",
        "retain": True,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
            request=request,
        )

    client = DeepSeekClient(
        "secret",
        ModelRouter(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert client.triage("title", "text").retain
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in captured
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["max_tokens"] == 1500


def test_pro_request_does_not_downgrade_on_failure() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(503, text="unavailable", request=request)

    client = DeepSeekClient(
        "secret",
        ModelRouter(),
        retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ValueError):
        client.call(AnalysisRole.ANALYST, TriageOutput, "Return JSON", "text")
    assert seen == ["deepseek-v4-pro"]
