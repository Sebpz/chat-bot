import anthropic

from app.core.config import get_settings
from app.interfaces.llm import ChatLLM
from app.utils.errors import LLMError

_MAX_TOKENS = 1024


class ClaudeChatLLM(ChatLLM):
    """ChatLLM backed by the Anthropic Claude API."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.anthropic_api_key
        self.model = model if model is not None else settings.claude_model
        self._client: anthropic.Anthropic | None = None

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APITimeoutError as exc:
            raise LLMError("Claude API timed out") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Claude API connection error: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error: {exc}") from exc

        return "".join(block.text for block in response.content if block.type == "text")
