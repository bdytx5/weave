from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import weave
from weave.integrations.patcher import MultiPatcher, NoOpPatcher, SymbolPatcher
from weave.trace.autopatch import IntegrationSettings, OpSettings
from weave.trace.op import _add_accumulator

if TYPE_CHECKING:
    from litellm.utils import ModelResponse

_litellm_patcher: MultiPatcher | None = None
_cache_initialized: bool = False


class DiskCache:
    """
    Drop-in replacement backend for LiteLLM cache.

    This wraps diskcache.Cache and provides both sync and async APIs
    that LiteLLM expects for its caching backend.
    """

    def __init__(self, directory, size_limit=None, underlying_cache=None):
        """
        Initialize disk cache.

        Args:
            directory: Path to cache directory
            size_limit: Optional size limit in bytes. If None, uses a very large default (2^62 bytes)
            underlying_cache: Optional existing diskcache.Cache instance to reuse
        """
        if underlying_cache is not None:
            # Reuse existing cache instance (shares storage with Weave cache)
            self._dc = underlying_cache
        else:
            # Create new cache instance
            import diskcache as dc
            cap = size_limit if size_limit is not None else (1 << 62)
            self._dc = dc.Cache(directory, size_limit=cap)

    # Sync API used by LiteLLM
    def get_cache(self, key, **kwargs):
        """Get value from cache by key."""
        result = self._dc.get(key)
        if result is not None:
            # Set context variable to indicate cache hit
            from weave.integrations.cache import _cache_hit
            _cache_hit.set(True)
        return result

    def set_cache(self, key, value, ttl=None, **kwargs):
        """Set value in cache with optional TTL."""
        expire = None if ttl is None else float(ttl)
        self._dc.set(key, value, expire=expire)

    # Async API used by LiteLLM
    async def async_get_cache(self, key, **kwargs):
        """Async get value from cache by key."""
        result = self.get_cache(key, **kwargs)
        # get_cache already sets _cache_hit if result is not None
        return result

    async def async_set_cache(self, key, value, ttl=None, **kwargs):
        """Async set value in cache with optional TTL."""
        return self.set_cache(key, value, ttl=ttl, **kwargs)

    async def async_set_cache_pipeline(self, cache_list, ttl=None, **kwargs):
        """
        Async batch set multiple cache entries.

        Args:
            cache_list: List of (key, value) tuples
            ttl: Optional time-to-live in seconds
        """
        for k, v in cache_list:
            self.set_cache(k, v, ttl=ttl)

    async def batch_cache_write(self, key, value, ttl=None, **kwargs):
        """Async batch write (single entry)."""
        self.set_cache(key, value, ttl=ttl)

    async def ping(self):
        """Async ping check."""
        return True

    async def delete_cache_keys(self, keys):
        """
        Async delete multiple cache keys.

        Args:
            keys: List of keys to delete
        """
        for k in keys:
            try:
                del self._dc[k]
            except KeyError:
                pass
        return True

    async def disconnect(self):
        """Async disconnect and close cache."""
        self._dc.close()




# This accumulator is nearly identical to the mistral accumulator, just with different types.
def litellm_accumulator(
    acc: ModelResponse | None,
    value: ModelResponse,
) -> ModelResponse:
    # This import should be safe at this point
    from litellm.utils import Choices, Message, ModelResponse, Usage

    if acc is None:
        acc = ModelResponse(
            id=value.id,
            object=value.object,
            created=value.created,
            model=value.model,
            choices=[],
            usage=Usage(prompt_tokens=0, total_tokens=0, completion_tokens=None),
        )

    # Merge in the usage info
    if "usage" in value.model_fields_set and value.usage is not None:
        acc.usage.prompt_tokens += value.usage.prompt_tokens
        acc.usage.total_tokens += value.usage.total_tokens
        if acc.usage.completion_tokens is None:
            acc.usage.completion_tokens = value.usage.completion_tokens
        else:
            acc.usage.completion_tokens += value.usage.completion_tokens

    # Loop through the choices and add their deltas
    for delta_choice in value.choices:
        while delta_choice.index >= len(acc.choices):
            acc.choices.append(
                Choices(
                    index=len(acc.choices),
                    message=Message(role="", content=""),
                    finish_reason=None,
                )
            )
        acc.choices[delta_choice.index].message.role = (
            delta_choice.delta.role or acc.choices[delta_choice.index].message.role
        )
        acc.choices[delta_choice.index].message.content += (
            delta_choice.delta.content or ""
        )
        if delta_choice.delta.tool_calls:
            if acc.choices[delta_choice.index].message.tool_calls is None:
                acc.choices[delta_choice.index].message.tool_calls = []
            acc.choices[
                delta_choice.index
            ].message.tool_calls += delta_choice.delta.tool_calls
        acc.choices[delta_choice.index].finish_reason = (
            delta_choice.finish_reason or acc.choices[delta_choice.index].finish_reason
        )

    return acc


# LiteLLM does so odd stuff with pydantic objects which result in our auto
# serialization not working correctly. Here we just blindly dump to a dict instead.
def litellm_on_finish_post_processor(value: Any) -> Any:
    import pydantic

    value_to_finish = value
    if isinstance(value, pydantic.BaseModel):
        value_to_finish = value.model_dump()

    return value_to_finish


# Unlike other integrations, streaming is based on input flag, not
def should_use_accumulator(inputs: dict) -> bool:
    return isinstance(inputs, dict) and bool(inputs.get("stream"))


def make_wrapper(settings: OpSettings) -> Callable:
    def litellm_wrapper(fn: Callable) -> Callable:
        from functools import wraps

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        op_kwargs = settings.model_dump()
        op = weave.op(wrapper, **op_kwargs)
        return _add_accumulator(
            op,  # type: ignore
            make_accumulator=lambda inputs: litellm_accumulator,
            should_accumulate=should_use_accumulator,
            on_finish_post_processor=litellm_on_finish_post_processor,
        )

    return litellm_wrapper


def get_litellm_patcher(
    settings: IntegrationSettings | None = None,
) -> MultiPatcher | NoOpPatcher:
    if settings is None:
        settings = IntegrationSettings()

    if not settings.enabled:
        return NoOpPatcher()

    global _litellm_patcher
    if _litellm_patcher is not None:
        return _litellm_patcher

    base = settings.op_settings

    completion_settings = base.model_copy(
        update={"name": base.name or "litellm.completion"}
    )
    acompletion_settings = base.model_copy(
        update={"name": base.name or "litellm.acompletion"}
    )

    _litellm_patcher = MultiPatcher(
        [
            SymbolPatcher(
                lambda: importlib.import_module("litellm"),
                "completion",
                make_wrapper(completion_settings),
            ),
            SymbolPatcher(
                lambda: importlib.import_module("litellm"),
                "acompletion",
                make_wrapper(acompletion_settings),
            ),
        ]
    )

    return _litellm_patcher
