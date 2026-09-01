from quant_research_radar.cli import _phase18_client
from quant_research_radar.config import Settings


def test_phase18_client_uses_only_configured_deepseek_provider() -> None:
    assert _phase18_client(Settings(llm_provider="fake")) is None

    client = _phase18_client(
        Settings(llm_provider="deepseek", deepseek_api_key="fixture-key")
    )

    assert client is not None
    assert client.provider == "deepseek"
