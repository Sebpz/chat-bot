from abc import ABC, abstractmethod


class ChatLLM(ABC):
    """Chat completion backend used for answer generation."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's answer text. Raises LLMError on failure/timeout."""
        raise NotImplementedError
