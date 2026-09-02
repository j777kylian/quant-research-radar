from quant_research_radar.cli import _phase18_client
from quant_research_radar.config import Settings


def test_phase18_client_uses_deepseek_credential_when_present() -> None:
    # No credential -> no client (critic stays honestly NOT_RUN).
    assert _phase18_client(Settings(deepseek_api_key=None)) is None

    # A credential enables the DeepSeek client regardless of the legacy
    # llm_provider flag, so the Methodology Critic can run unattended.
    client = _phase18_client(
        Settings(llm_provider="fake", deepseek_api_key="fixture-key")
    )

    assert client is not None
    assert client.provider == "deepseek"
