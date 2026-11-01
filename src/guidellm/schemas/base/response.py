"""
Backend response models for request and response handling.

Provides standardized response models for generation operations that capture
output text, usage metrics, and compilation of request statistics. Ensures
consistent data handling and statistics aggregation across different backend
implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import Field

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from guidellm.schemas.base.base import StandardBaseModel
from guidellm.schemas.base.info import RequestInfo
from guidellm.schemas.base.request import GenerationRequest, UsageMetrics
from guidellm.schemas.base.request_stats import GenerativeRequestStats
from guidellm.schemas.base.tool_call import ToolCall, ToolCallFunction

OutputTokenizerFactory = Callable[[], "PreTrainedTokenizerBase"]

__all__ = [
    "GenerationResponse",
    "OutputTokenizerFactory",
    "ToolCall",
    "ToolCallFunction",
]


class GenerationResponse(StandardBaseModel):
    """
    Response model for backend generation operations.

    Captures the output and metrics from a generation request, providing structured
    data for text output, tool call payloads, token usage statistics, and compilation
    of detailed request statistics for analysis and monitoring purposes.

    Example:
    ::
        response = GenerationResponse(
            request_id="req-123",
            text="Generated response text",
            input_metrics=UsageMetrics(token_count=50),
            output_metrics=UsageMetrics(token_count=25)
        )
        stats = response.compile_stats(request, info)
    """

    request_id: str = Field(
        description="Unique identifier matching the original GenerationRequest."
    )
    response_id: str | None = Field(
        default=None,
        description="Unique identifier matching the original vLLM Response ID.",
    )
    request_args: str | None = Field(
        description="Arguments passed to the backend for request processing."
    )
    text: str | None = Field(
        default=None,
        description="The generated response text.",
    )
    reasoning_text: str | None = Field(
        default=None,
        description=(
            "Reasoning/chain-of-thought text emitted before the content. "
            "Not included in multi-turn history unless explicitly enabled."
        ),
    )
    tool_calls: list[ToolCall] | None = Field(
        default=None,
        description=(
            "Raw tool call payloads from the model response, each containing "
            "id, type, and function (name + arguments) in OpenAI format."
        ),
    )
    input_metrics: UsageMetrics = Field(
        default_factory=UsageMetrics,
        description="Token usage statistics from the input prompt.",
    )
    output_metrics: UsageMetrics = Field(
        default_factory=UsageMetrics,
        description="Token usage statistics from the generated output.",
    )
    response_metrics: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Per-request metrics the backend reported alongside usage, passed through "
            "as received. vLLM populates this with its `metrics` object, which carries "
            "server-side timings and speculative-decoding acceptance."
        ),
    )

    def compile_stats(
        self,
        request: GenerationRequest,
        info: RequestInfo,
        prefer_response: bool = True,
        tokenizer_factory: OutputTokenizerFactory | None = None,
    ) -> GenerativeRequestStats:
        """
        Compile and return comprehensive request statistics.

        Merges metrics from the request and response objects to create a complete
        statistical record, with preference given to response-level metrics when
        available to ensure accuracy of actual execution data.

        When the response did not report an output token count,
        ``tokenizer_factory`` counts tokens in the response text. That count
        replaces the request target, which is the length the client asked for.

        :param request: The original generation request containing input data
        :param info: Metadata and timing information for the request execution
        :param prefer_response: Whether to prefer response metrics over request
            metrics when both are available
        :param tokenizer_factory: Factory for the tokenizer that counts response
            text when the response has no output token count. When omitted, the
            request target is kept.
        :return: A GenerativeRequestStats object containing detailed statistics
        :raises ValueError: When request IDs don't match between objects
        """
        if request.request_id != self.request_id:
            raise ValueError("Mismatched request IDs between request and response.")

        if info.request_id != self.request_id:
            raise ValueError("Mismatched request IDs between info and response.")

        if info.status != "completed":
            # Clear out request output metrics if the request failed since
            # those are not valid.  Preserve tool_call_count=0 when the
            # request expected a tool call so errored-request statistics
            # still contribute to the tool call metrics table.
            is_tool_call_turn = request.turn_type == "client_tool_call"
            request.output_metrics = UsageMetrics()
            if is_tool_call_turn:
                request.output_metrics.tool_call_count = 0

        base_input = request.input_metrics if prefer_response else self.input_metrics
        override_input = (
            self.input_metrics if prefer_response else request.input_metrics
        )
        base_output = request.output_metrics if prefer_response else self.output_metrics
        override_output = (
            self.output_metrics if prefer_response else request.output_metrics
        )

        input_metrics_dict = base_input.model_dump()
        for key, value in override_input.model_dump().items():
            if value is not None:
                input_metrics_dict[key] = value
        output_metrics_dict = base_output.model_dump()
        for key, value in override_output.model_dump().items():
            if value is not None:
                output_metrics_dict[key] = value

        self._apply_counted_output_tokens(
            output_metrics_dict, info, prefer_response, tokenizer_factory
        )

        # If tool calls were expected but none were recorded (either because
        # the request errored or the model produced none with ignore_continue),
        # set tool_call_count=0 so the metrics table still appears.
        if (
            request.turn_type == "client_tool_call"
            and output_metrics_dict.get("tool_call_count") is None
        ):
            output_metrics_dict["tool_call_count"] = 0

        return GenerativeRequestStats(
            request_id=self.request_id,
            response_id=self.response_id,
            request_args=self.request_args,
            output=self.text,
            reasoning_output=self.reasoning_text,
            tool_calls=self.tool_calls,
            info=info,
            input_metrics=UsageMetrics(**input_metrics_dict),
            output_metrics=UsageMetrics(**output_metrics_dict),
            response_metrics=self.response_metrics,
        )

    def _apply_counted_output_tokens(
        self,
        output_metrics: dict[str, Any],
        info: RequestInfo,
        prefer_response: bool,
        tokenizer_factory: OutputTokenizerFactory | None,
    ) -> None:
        """
        Replace a missing output token count with a count of the response text.

        The factory is called only when response metrics are preferred, the
        request completed, usage did not report an output token count, and the
        response has text.

        :param output_metrics: Merged output metrics to update in place
        :param info: Request execution information
        :param prefer_response: Whether response metrics are preferred
        :param tokenizer_factory: Factory for the tokenizer, or None
        """
        text = self.text
        if (
            tokenizer_factory is None
            or text is None
            or not prefer_response
            or info.status != "completed"
            or self.output_metrics.text_tokens is not None
        ):
            return
        output_metrics["text_tokens"] = len(
            tokenizer_factory().encode(text, add_special_tokens=False)
        )
