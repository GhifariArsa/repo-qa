"""Provider abstraction for the question naturalizer.

The naturalizer only ever asks a model for one thing: a structured rewrite of a
question. That is a small enough surface to keep provider-neutral, so the same
prompts, validators, and judge run against either Anthropic or OpenAI.

Keeping both is not only convenience. If the benchmark is later evaluated on
Claude, having the questions *written* by a different model family removes an
obvious self-preference confound — the phrasing wasn't tuned by the same model
being scored. Record which provider wrote each question (the naturalizer does)
so that can be reported.

Model spec format: ``<provider>:<model>``, e.g. ``anthropic:claude-opus-5`` or
``openai:gpt-4.1``. A bare model id starting with ``claude`` is treated as
Anthropic; anything else bare is ambiguous and must carry a prefix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"


class LLMProvider(Protocol):
    """Returns a validated instance of `schema`, or None if the model refused."""

    name: str
    model: str

    def parse(self, system: str, user: str, schema: type[T], max_tokens: int) -> T | None: ...


@dataclass
class AnthropicProvider:
    model: str = DEFAULT_ANTHROPIC_MODEL
    client: Any = None
    name: str = "anthropic"

    def __post_init__(self):
        if self.client is None:
            import anthropic

            # Resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
            # `ant auth login` profile — do not pass a key explicitly.
            self.client = anthropic.Anthropic()

    def parse(self, system: str, user: str, schema: type[T], max_tokens: int) -> T | None:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=max_tokens,
            # The system prompt is byte-identical across every question, so it
            # is the right cache prefix boundary; per-question content follows.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        if getattr(response, "stop_reason", None) == "refusal":
            return None
        return response.parsed_output


@dataclass
class OpenAIProvider:
    model: str
    client: Any = None
    name: str = "openai"

    def __post_init__(self):
        if self.client is None:
            import openai

            self.client = openai.OpenAI()   # resolves OPENAI_API_KEY

    def parse(self, system: str, user: str, schema: type[T], max_tokens: int) -> T | None:
        completion = self.client.chat.completions.parse(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
        )
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            return None
        return message.parsed


def parse_model_spec(spec: str) -> tuple[str, str]:
    """`"openai:gpt-4.1"` -> `("openai", "gpt-4.1")`."""
    if ":" in spec:
        provider, _, model = spec.partition(":")
        provider = provider.strip().lower()
        if provider not in ("anthropic", "openai"):
            raise SystemExit(f"unknown provider {provider!r}; use 'anthropic' or 'openai'")
        if not model.strip():
            raise SystemExit(f"model spec {spec!r} has no model id after the provider")
        return provider, model.strip()
    if spec.startswith("claude"):
        return "anthropic", spec
    raise SystemExit(
        f"ambiguous model spec {spec!r} — prefix it with the provider, "
        f"e.g. 'anthropic:{spec}' or 'openai:{spec}'"
    )


def _require_credentials(provider: str) -> None:
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set")
        return
    # Anthropic resolves a key OR an `ant auth login` profile, so an unset
    # env var is not by itself an error — only warn when neither is present.
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return
    config_dir = os.environ.get("ANTHROPIC_CONFIG_DIR") or os.path.expanduser("~/.config/anthropic")
    if not os.path.isdir(config_dir):
        raise SystemExit(
            "no Anthropic credentials found: set ANTHROPIC_API_KEY, or run `ant auth login`"
        )


def get_provider(spec: str, client: Any = None) -> LLMProvider:
    provider, model = parse_model_spec(spec)
    if client is None:
        _require_credentials(provider)
    if provider == "openai":
        return OpenAIProvider(model=model, client=client)
    return AnthropicProvider(model=model, client=client)
