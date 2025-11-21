"""The top-level functions and classes for working with Weave."""

from weave import version
from weave.trace.api import *

__version__ = version.VERSION

from weave.integrations.wandb import wandb_init_hook

wandb_init_hook()

# Import cache utilities
from weave.integrations import cache as _cache_module

from weave.agent.agent import Agent as Agent
from weave.agent.agent import AgentState as AgentState
from weave.dataset.dataset import Dataset
from weave.evaluation.eval import Evaluation
from weave.evaluation.eval_imperative import EvaluationLogger
from weave.flow.annotation_spec import AnnotationSpec
from weave.flow.model import Model
from weave.flow.monitor import Monitor
from weave.flow.saved_view import SavedView
from weave.flow.scorer import Scorer
from weave.initialization import *
from weave.object.obj import Object
from weave.prompt.prompt import EasyPrompt, MessagesPrompt, Prompt, StringPrompt
from weave.trace.util import Thread as Thread
from weave.trace.util import ThreadPoolExecutor as ThreadPoolExecutor
from weave.type_handlers.Audio.audio import Audio
from weave.type_handlers.File.file import File
from weave.type_handlers.Markdown.markdown import Markdown
from weave.type_wrappers import Content

# Alias for succinct code
P = EasyPrompt


# Cache namespace with convenient access
class cache:
    """LLM caching utilities.

    Examples:
        # Enable/disable cache globally
        weave.cache.enable()
        weave.cache.disable()

        # Disable cache for a block
        with weave.disable_cache():
            result = model.generate(...)

        # Clear all cache
        weave.cache.clear()

        # Clear by provider
        weave.cache.clear(provider="openai")

        # Clear by model pattern
        weave.cache.clear(model="gpt-4*")
    """

    @staticmethod
    def enable() -> None:
        """Enable LLM caching globally."""
        _cache_module.enable_cache()

    @staticmethod
    def disable() -> None:
        """Disable LLM caching globally."""
        _cache_module.disable_cache_globally()

    @staticmethod
    def clear(provider: str = None, model: str = None) -> int:
        """Clear cache entries, optionally filtered by provider or model.

        Args:
            provider: Optional provider name (e.g., "openai", "anthropic")
            model: Optional model pattern (supports wildcards like "gpt-4*")

        Returns:
            Number of entries cleared
        """
        cache_instance = _cache_module.get_global_cache()
        if cache_instance is None:
            return 0
        return cache_instance.clear(provider=provider, model=model)

    @staticmethod
    def stats() -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with size_limit, current_size, item_count, and percent_full
        """
        cache_instance = _cache_module.get_global_cache()
        if cache_instance is None:
            return {}
        return cache_instance.get_stats()

    @staticmethod
    def print_stats() -> None:
        """Print human-readable cache statistics."""
        cache_instance = _cache_module.get_global_cache()
        if cache_instance is not None:
            cache_instance.print_stats()


# Re-export disable_cache context manager at top level
disable_cache = _cache_module.disable_cache

# Special object informing doc generation tooling which symbols
# to document & to associate with this module.
__docspec__ = [
    # Re-exported from trace.api
    init,
    publish,
    ref,
    get,
    require_current_call,
    get_current_call,
    finish,
    op,
    attributes,
    thread,
    # Re-exported from flow module
    Object,
    Dataset,
    Model,
    Prompt,
    StringPrompt,
    MessagesPrompt,
    Evaluation,
    EvaluationLogger,
    Scorer,
    AnnotationSpec,
    File,
    Content,
    Markdown,
    Monitor,
    SavedView,
    Audio,
]

__all__ = [
    "Agent",
    "AgentState",
    "AnnotationSpec",
    "Audio",
    "Content",
    "Dataset",
    "EasyPrompt",
    "Evaluation",
    "EvaluationLogger",
    "File",
    "Markdown",
    "MessagesPrompt",
    "Model",
    "Monitor",
    "Object",
    "Prompt",
    "SavedView",
    "Scorer",
    "StringPrompt",
    "attributes",
    "cache",
    "disable_cache",
    "finish",
    "get",
    "get_current_call",
    "init",
    "op",
    "publish",
    "ref",
    "require_current_call",
    "set_view",
    "thread",
]
