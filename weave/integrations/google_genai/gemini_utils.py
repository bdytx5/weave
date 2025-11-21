from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional

import weave
from weave.trace.autopatch import OpSettings
from weave.trace.call import Call
from weave.trace.op import _add_accumulator
from weave.trace.serialization.serialize import dictify

if TYPE_CHECKING:
    from google.genai.types import GenerateContentResponse

SKIP_TRACING_FUNCTIONS = [
    "google.genai.models.Models.count_tokens",
    "google.genai.models.AsyncModels.count_tokens",
]


def maybe_unwrap_google_genai_response(value: Any) -> Any:
    """Unwrap Google GenAI response for caching."""
    # Google GenAI responses are pydantic models, return as-is for serialization
    print(f"[GOOGLE GENAI UNWRAP] Type: {type(value)}")
    return value


def maybe_wrap_google_genai_response(value: Any) -> Any:
    """Reconstruct Google GenAI response objects from cached dicts."""
    print(f"[GOOGLE GENAI WRAP] Input type: {type(value)}, is_dict: {isinstance(value, dict)}")
    if not isinstance(value, dict):
        print(f"[GOOGLE GENAI WRAP] Not a dict, returning as-is")
        return value

    try:
        from google.genai.types import GenerateContentResponse

        # Try to reconstruct GenerateContentResponse from dict
        if "candidates" in value or "usage_metadata" in value:
            try:
                result = GenerateContentResponse(**value)
                print(f"[GOOGLE GENAI WRAP] Successfully reconstructed to {type(result)}")
                return result
            except Exception as e:
                print(f"[GOOGLE GENAI WRAP] Reconstruction failed: {e}")
                pass
    except Exception as e:
        print(f"[GOOGLE GENAI WRAP] Import/check failed: {e}")
        pass

    print(f"[GOOGLE GENAI WRAP] Returning dict as-is")
    return value


def google_genai_gemini_postprocess_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Postprocess inputs of the trace for the Google GenAI Gemini API to be used in
    the trace visualization in the Weave UI. If the parameter `self` is present
    (i.e, if the function being traced is a stateful method), it is converted to a
    dictionary of attributes that can be displayed in the Weave UI.
    """
    # Extract the model name from the inputs and ensure it is present in the inputs
    # First check if model is already in inputs as a kwarg
    if "model" not in inputs and "self" in inputs:
        model_name = getattr(inputs["self"], "_model", None)
        if model_name is not None:
            inputs["model"] = model_name

    # Convert the `self` parameter which is actually the state of the
    # `google.genai.models.Models` object to a dictionary of attributes that can
    # be displayed in the Weave UI
    if "self" in inputs:
        inputs["self"] = dictify(inputs["self"])
    return inputs


def google_genai_gemini_on_finish(
    call: Call, output: Any, exception: Optional[BaseException] = None
) -> None:
    """On finish handler for the Google GenAI Gemini API integration that ensures the usage
    metadata is added to the summary of the trace.
    """
    model_name = call.inputs.get("model")
    if not model_name:
        # Model might not be extracted, skip usage tracking
        return
    usage = {model_name: {"requests": 1}}
    summary_update = {"usage": usage}
    if output:
        call.output = dictify(output)
        if hasattr(output, "usage_metadata"):
            usage[model_name].update(
                {
                    "prompt_tokens": output.usage_metadata.prompt_token_count,
                    "completion_tokens": output.usage_metadata.candidates_token_count,
                    "total_tokens": output.usage_metadata.total_token_count,
                }
            )
    if call.summary is not None:
        call.summary.update(summary_update)


def google_genai_gemini_accumulator(
    acc: Optional["GenerateContentResponse"], value: "GenerateContentResponse"
) -> "GenerateContentResponse":
    if acc is None:
        return value

    for i, value_candidate in enumerate(value.candidates):
        if i >= len(acc.candidates):
            break
        for j, value_part in enumerate(value_candidate.content.parts):
            if j >= len(acc.candidates[i].content.parts):
                break
            if value_part.text is not None:
                acc.candidates[i].content.parts[j].text += value_part.text

    if acc.usage_metadata.prompt_token_count is None:
        acc.usage_metadata.prompt_token_count = 0
    elif value.usage_metadata.prompt_token_count is not None:
        acc.usage_metadata.prompt_token_count += value.usage_metadata.prompt_token_count

    if acc.usage_metadata.candidates_token_count is None:
        acc.usage_metadata.candidates_token_count = 0
    elif value.usage_metadata.candidates_token_count is not None:
        acc.usage_metadata.candidates_token_count += (
            value.usage_metadata.candidates_token_count
        )

    if acc.usage_metadata.total_token_count is None:
        acc.usage_metadata.total_token_count = 0
    elif value.usage_metadata.total_token_count is not None:
        acc.usage_metadata.total_token_count += value.usage_metadata.total_token_count

    if acc.usage_metadata.cached_content_token_count is None:
        acc.usage_metadata.cached_content_token_count = 0
    elif value.usage_metadata.cached_content_token_count is not None:
        acc.usage_metadata.cached_content_token_count += (
            value.usage_metadata.cached_content_token_count
        )

    return acc


def google_genai_gemini_wrapper_sync(
    settings: OpSettings,
) -> Callable[[Callable], Callable]:
    def wrapper(fn: Callable) -> Callable:
        from weave.integrations.cache import with_llm_cache

        @with_llm_cache("google_genai", unwrap_fn=maybe_unwrap_google_genai_response, wrap_fn=maybe_wrap_google_genai_response)
        @wraps(fn)
        def _cached_fn(self, *args, **kwargs):
            result = fn(self, *args, **kwargs)
            return result

        op_kwargs = settings.model_dump()
        if not op_kwargs.get("postprocess_inputs"):
            op_kwargs["postprocess_inputs"] = google_genai_gemini_postprocess_inputs

        op = weave.op(_cached_fn, **op_kwargs)
        if op.name not in SKIP_TRACING_FUNCTIONS:
            op._set_on_finish_handler(google_genai_gemini_on_finish)
        return _add_accumulator(
            op,
            make_accumulator=lambda inputs: google_genai_gemini_accumulator,
            should_accumulate=lambda inputs: op.name.endswith("stream"),
        )

    return wrapper


def google_genai_gemini_wrapper_async(
    settings: OpSettings,
) -> Callable[[Callable], Callable]:
    def wrapper(fn: Callable) -> Callable:
        from weave.integrations.cache import with_llm_cache

        @wraps(fn)
        @with_llm_cache("google_genai", unwrap_fn=maybe_unwrap_google_genai_response, wrap_fn=maybe_wrap_google_genai_response)
        async def _async_cached_fn(self, *args: Any, **kwargs: Any) -> Any:
            return await fn(self, *args, **kwargs)

        op_kwargs = settings.model_dump()
        if not op_kwargs.get("postprocess_inputs"):
            op_kwargs["postprocess_inputs"] = google_genai_gemini_postprocess_inputs

        op = weave.op(_async_cached_fn, **op_kwargs)
        if op.name not in SKIP_TRACING_FUNCTIONS:
            op._set_on_finish_handler(google_genai_gemini_on_finish)
        return _add_accumulator(
            op,
            make_accumulator=lambda inputs: google_genai_gemini_accumulator,
            should_accumulate=lambda inputs: op.name.endswith("stream"),
        )

    return wrapper
