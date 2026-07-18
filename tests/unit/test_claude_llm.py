from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.implementations.claude_llm import ClaudeChatLLM
from app.utils.errors import LLMError


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


@patch("app.implementations.claude_llm.anthropic.Anthropic")
def test_generate_calls_claude_with_temperature_zero(mock_anthropic_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = MagicMock(content=[_text_block("The answer is 42.")])

    llm = ClaudeChatLLM(api_key="test-key", model="claude-sonnet-5")
    result = llm.generate("system prompt", "user question")

    assert result == "The answer is 42."
    mock_client.messages.create.assert_called_once()
    _, kwargs = mock_client.messages.create.call_args
    assert kwargs["temperature"] == 0
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["system"] == "system prompt"
    assert kwargs["messages"] == [{"role": "user", "content": "user question"}]


@patch("app.implementations.claude_llm.anthropic.Anthropic")
def test_generate_concatenates_multiple_text_blocks(mock_anthropic_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = MagicMock(
        content=[_text_block("Part one. "), _text_block("Part two.")]
    )

    llm = ClaudeChatLLM(api_key="test-key")
    result = llm.generate("system", "question")

    assert result == "Part one. Part two."


@patch("app.implementations.claude_llm.anthropic.Anthropic")
def test_generate_wraps_timeout_as_llm_error(mock_anthropic_cls: MagicMock) -> None:
    import anthropic

    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    mock_client.messages.create.side_effect = anthropic.APITimeoutError(request=request)

    llm = ClaudeChatLLM(api_key="test-key")

    with pytest.raises(LLMError):
        llm.generate("system", "question")
