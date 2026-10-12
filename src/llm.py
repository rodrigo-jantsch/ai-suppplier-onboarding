"""One interface to the language models, whichever provider is behind them.

Every other module calls chat_json / chat_text / embed, never a provider SDK
directly. Switching between Gemini (hosted, free tier) and Ollama (local) is
one line in config/settings.yaml, and comparing providers on the same gold
set makes a good result for the README.
"""
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.config import settings

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------- providers

def _gemini_generate(prompt: str, system: str | None, schema: dict | None) -> str:
    """One Gemini call. Reads GEMINI_API_KEY from the environment (.env)."""
    from google import genai

    cfg = settings()["llm"]
    client = genai.Client()
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    kwargs = {"model": cfg["chat_model"], "input": full_prompt}
    if schema is not None:
        kwargs["response_format"] = {"type": "text", "mime_type": "application/json", "schema": schema}
    interaction = client.interactions.create(**kwargs)
    return interaction.output_text


def _ollama_generate(prompt: str, system: str | None, schema: dict | None) -> str:
    import ollama

    cfg = settings()["llm"]
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    kwargs = {
        "model": cfg["ollama_chat_model"],
        "messages": messages,
        "options": {"temperature": cfg.get("temperature", 0)},
    }
    if schema is not None:
        kwargs["format"] = schema
    return ollama.chat(**kwargs).message.content


_PROVIDERS = {"gemini": _gemini_generate, "ollama": _ollama_generate}


def _generate(prompt: str, system: str | None = None, schema: dict | None = None) -> str:
    provider = settings()["llm"]["provider"]
    if provider not in _PROVIDERS:
        raise NotImplementedError(f"Provider '{provider}' not wired up yet")
    return _PROVIDERS[provider](prompt, system, schema)


# ---------------------------------------------------------------- public API

def chat_text(prompt: str, system: str | None = None) -> str:
    """Plain text reply, for quick tests and free-form steps."""
    return _generate(prompt, system)


def chat_json(prompt: str, schema: type[T], system: str | None = None) -> T:
    """Ask the model for JSON matching `schema` and return a validated object.

    The schema is sent to the model to constrain its output, and the reply is
    validated again here. If validation fails, retry once with the error.
    """
    json_schema = schema.model_json_schema()
    attempt_prompt = prompt
    last_error: Exception | None = None
    for _ in range(settings()["llm"].get("max_retries", 1) + 1):
        content = _generate(attempt_prompt, system, json_schema)
        try:
            return schema.model_validate_json(content)
        except ValidationError as err:
            last_error = err
            attempt_prompt = (
                f"{prompt}\n\nYour previous answer failed validation:\n{err}\n"
                f"Previous answer:\n{content}\nReturn corrected JSON only."
            )
    raise ValueError(f"Model output failed validation after retries: {last_error}")


def embed(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per input text."""
    cfg = settings()["embeddings"]
    if cfg["provider"] != "ollama":
        raise NotImplementedError(f"Embedding provider '{cfg['provider']}' not wired up yet")
    import ollama

    return ollama.embed(model=cfg["model"], input=texts).embeddings
