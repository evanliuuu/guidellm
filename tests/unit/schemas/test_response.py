"""
Unit tests for GenerationResponse.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from guidellm.backends.openai.request_handlers import ChatCompletionsRequestHandler
from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
    GenerationResponse,
    GenerativeRequestStats,
    RequestInfo,
    RequestTimings,
    StandardBaseModel,
    ToolCall,
    ToolCallFunction,
    UsageMetrics,
)


class TestGenerationResponse:
    """Test cases for GenerationResponse model."""

    @pytest.fixture(
        params=[
            {
                "request_id": "test-123",
                "request_args": "model=gpt-3.5-turbo",
            },
            {
                "request_id": "test-456",
                "request_args": "model=gpt-4",
                "text": "Generated text",
            },
            {
                "request_id": "test-789",
                "request_args": None,
                "text": "Another response",
                "input_metrics": UsageMetrics(text_tokens=50),
                "output_metrics": UsageMetrics(text_tokens=25),
            },
        ],
        ids=["minimal", "with_text", "with_metrics"],
    )
    def valid_instances(self, request):
        """Fixture providing valid GenerationResponse instances."""
        constructor_args = request.param
        instance = GenerationResponse(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test GenerationResponse inheritance and type relationships."""
        assert issubclass(GenerationResponse, StandardBaseModel)
        assert hasattr(GenerationResponse, "model_dump")
        assert hasattr(GenerationResponse, "model_validate")

        # Check all expected fields and properties are defined
        fields = GenerationResponse.model_fields
        expected_fields = [
            "request_id",
            "request_args",
            "text",
            "input_metrics",
            "output_metrics",
        ]
        for field in expected_fields:
            assert field in fields

        # Check methods exist
        assert hasattr(GenerationResponse, "compile_stats")

    @pytest.mark.smoke
    def test_initialization(
        self, valid_instances: tuple[GenerationResponse, dict[str, str]]
    ):
        """Test GenerationResponse initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, GenerationResponse)
        assert instance.request_id == constructor_args["request_id"]
        assert instance.request_args == constructor_args["request_args"]

        # Check defaults for optional fields
        if "text" not in constructor_args:
            assert instance.text is None
        else:
            assert instance.text == constructor_args["text"]

        # Check metrics (either provided or default)
        assert hasattr(instance, "input_metrics")
        assert hasattr(instance, "output_metrics")
        assert isinstance(instance.input_metrics, UsageMetrics)
        assert isinstance(instance.output_metrics, UsageMetrics)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("request_id", None),
            ("request_id", 123),
            ("request_id", {}),
            ("request_args", 123),
            ("request_args", []),
            ("text", 123),
            ("text", []),
            ("input_metrics", "invalid"),
            ("output_metrics", "invalid"),
        ],
    )
    def test_invalid_initialization_values(self, field: str, value: Any):
        """Test GenerationResponse with invalid field values."""
        data = {"request_id": "test-id", "request_args": "test_args"}
        data[field] = value
        with pytest.raises(ValidationError):
            GenerationResponse(**data)

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test GenerationResponse initialization without required fields."""
        with pytest.raises(ValidationError):
            GenerationResponse()

        with pytest.raises(ValidationError):
            GenerationResponse(request_id="test")

        with pytest.raises(ValidationError):
            GenerationResponse(request_args="test")

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("prefer_response", "expected_tokens"),
        [
            (True, 75),
            (False, 60),
        ],
    )
    def test_compile_stats(self, prefer_response: bool, expected_tokens: int):
        """Test compile_stats method functionality."""
        response = GenerationResponse(
            request_id="test-123",
            request_args="test_args",
            text="Generated response",
            input_metrics=UsageMetrics(text_tokens=50),
            output_metrics=UsageMetrics(text_tokens=25),
        )

        request = GenerationRequest(
            request_id="test-123",
            input_metrics=UsageMetrics(text_tokens=40),
            output_metrics=UsageMetrics(text_tokens=20),
        )

        request_info = RequestInfo(request_id="test-123", status="completed")

        stats = response.compile_stats(request, request_info, prefer_response)
        assert isinstance(stats, GenerativeRequestStats)
        assert stats.request_id == "test-123"
        assert stats.output == "Generated response"

        total_tokens = (
            stats.input_metrics.text_tokens + stats.output_metrics.text_tokens
        )
        assert total_tokens == expected_tokens

    @pytest.mark.smoke
    def test_compile_stats_with_defaults(self):
        """Test compile_stats with default metrics."""
        response = GenerationResponse(
            request_id="test-123",
            request_args="test_args",
            text="Generated response",
        )

        request = GenerationRequest(
            request_id="test-123",
        )

        request_info = RequestInfo(request_id="test-123", status="completed")

        stats = response.compile_stats(request, request_info)
        assert isinstance(stats, GenerativeRequestStats)
        assert stats.request_id == "test-123"
        assert stats.output == "Generated response"

    @pytest.mark.smoke
    def test_compile_stats_failed_request(self):
        """Test compile_stats with failed request status."""
        response = GenerationResponse(
            request_id="test-123",
            request_args="test_args",
            text=None,
            output_metrics=UsageMetrics(text_tokens=25),
        )

        request = GenerationRequest(
            request_id="test-123",
            output_metrics=UsageMetrics(text_tokens=20),
        )

        request_info = RequestInfo(request_id="test-123", status="errored")

        stats = response.compile_stats(request, request_info)
        assert isinstance(stats, GenerativeRequestStats)
        assert stats.request_id == "test-123"
        assert stats.info.status == "errored"

    @pytest.mark.smoke
    def test_compile_stats_errored_preserves_tool_call_count_when_expected(self):
        """
        compile_stats sets tool_call_count=0 on errored requests that
        expected tool calls, so the tool call metrics table still appears.

        ## WRITTEN BY AI ##
        """
        response = GenerationResponse(
            request_id="tc-err",
            request_args="test_args",
            text=None,
            output_metrics=UsageMetrics(text_tokens=10),
        )
        request = GenerationRequest(
            request_id="tc-err",
            turn_type="client_tool_call",
            output_metrics=UsageMetrics(text_tokens=20, tool_call_count=3),
        )
        info = RequestInfo(request_id="tc-err", status="errored")

        stats = response.compile_stats(request, info)
        assert stats.output_metrics.tool_call_count == 0

    @pytest.mark.smoke
    def test_compile_stats_errored_no_tool_call_count_when_not_expected(self):
        """
        compile_stats leaves tool_call_count as None on errored requests
        that did not expect tool calls.

        ## WRITTEN BY AI ##
        """
        response = GenerationResponse(
            request_id="no-tc-err",
            request_args="test_args",
            text=None,
            output_metrics=UsageMetrics(text_tokens=10),
        )
        request = GenerationRequest(
            request_id="no-tc-err",
            turn_type="standard",
            output_metrics=UsageMetrics(text_tokens=20),
        )
        info = RequestInfo(request_id="no-tc-err", status="errored")

        stats = response.compile_stats(request, info)
        assert stats.output_metrics.tool_call_count is None

    @pytest.mark.smoke
    def test_compile_stats_completed_sets_tool_call_count_zero_when_expected(self):
        """
        compile_stats sets tool_call_count=0 on completed requests where
        tool calls were expected but the model produced none (e.g.
        ignore_continue with tool_choice=none).

        ## WRITTEN BY AI ##
        """
        response = GenerationResponse(
            request_id="tc-ok-none",
            request_args="test_args",
            text="plain text response",
            output_metrics=UsageMetrics(text_tokens=15),
        )
        request = GenerationRequest(
            request_id="tc-ok-none",
            turn_type="client_tool_call",
            output_metrics=UsageMetrics(text_tokens=20),
        )
        info = RequestInfo(request_id="tc-ok-none", status="completed")

        stats = response.compile_stats(request, info)
        assert stats.output_metrics.tool_call_count == 0

    @pytest.mark.smoke
    def test_compile_stats_persists_tool_calls(self):
        """
        compile_stats carries tool_calls from the response to the stats.

        ## WRITTEN BY AI ##
        """
        tool_calls = [
            ToolCall(
                id="call_1",
                function=ToolCallFunction(
                    name="get_weather", arguments='{"city": "NYC"}'
                ),
            )
        ]
        response = GenerationResponse(
            request_id="test-tc",
            request_args="test_args",
            text=None,
            tool_calls=tool_calls,
        )

        request = GenerationRequest(request_id="test-tc")
        info = RequestInfo(request_id="test-tc", status="completed")

        stats = response.compile_stats(request, info)
        assert stats.tool_calls == tool_calls

    @pytest.mark.smoke
    def test_compile_stats_tool_calls_none_when_absent(self):
        """
        compile_stats leaves tool_calls as None for plain-text responses.

        ## WRITTEN BY AI ##
        """
        response = GenerationResponse(
            request_id="test-plain",
            request_args="test_args",
            text="Hello world",
        )

        request = GenerationRequest(request_id="test-plain")
        info = RequestInfo(request_id="test-plain", status="completed")

        stats = response.compile_stats(request, info)
        assert stats.tool_calls is None

    @pytest.mark.smoke
    def test_compile_stats_persists_reasoning_text(self):
        """
        compile_stats carries reasoning_text to reasoning_output on stats.

        ## WRITTEN BY AI ##
        """
        response = GenerationResponse(
            request_id="test-reason",
            request_args=None,
            text="Answer is 42",
            reasoning_text="Let me think step by step.",
        )
        request = GenerationRequest(request_id="test-reason")
        info = RequestInfo(request_id="test-reason", status="completed")

        stats = response.compile_stats(request, info)
        assert stats.output == "Answer is 42"
        assert stats.reasoning_output == "Let me think step by step."

    @pytest.mark.smoke
    def test_compile_stats_reasoning_output_none_when_absent(self):
        """
        compile_stats leaves reasoning_output as None when no reasoning
        was emitted.

        ## WRITTEN BY AI ##
        """
        response = GenerationResponse(
            request_id="test-no-reason",
            request_args=None,
            text="Plain answer",
        )
        request = GenerationRequest(request_id="test-no-reason")
        info = RequestInfo(request_id="test-no-reason", status="completed")

        stats = response.compile_stats(request, info)
        assert stats.output == "Plain answer"
        assert stats.reasoning_output is None

    @pytest.mark.sanity
    def test_compile_stats_mismatched_request_id(self):
        """Test compile_stats with mismatched request IDs."""
        response = GenerationResponse(
            request_id="test-123",
            request_args="test_args",
        )

        request = GenerationRequest(
            request_id="test-456",
        )

        request_info = RequestInfo(request_id="test-123")

        with pytest.raises(ValueError, match="Mismatched request IDs"):
            response.compile_stats(request, request_info)

    @pytest.mark.sanity
    def test_compile_stats_mismatched_info_id(self):
        """Test compile_stats with mismatched info request ID."""
        response = GenerationResponse(
            request_id="test-123",
            request_args="test_args",
        )

        request = GenerationRequest(
            request_id="test-123",
        )

        request_info = RequestInfo(request_id="test-456")

        with pytest.raises(ValueError, match="Mismatched request IDs"):
            response.compile_stats(request, request_info)

    @pytest.mark.sanity
    def test_marshalling(
        self, valid_instances: tuple[GenerationResponse, dict[str, str]]
    ):
        """Test GenerationResponse serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict["request_id"] == constructor_args["request_id"]
        assert data_dict["request_args"] == constructor_args["request_args"]

        # Test reconstruction
        reconstructed = GenerationResponse.model_validate(data_dict)
        assert reconstructed.request_id == instance.request_id
        assert reconstructed.request_args == instance.request_args
        assert reconstructed.text == instance.text
        assert (
            reconstructed.input_metrics.model_dump()
            == instance.input_metrics.model_dump()
        )
        assert (
            reconstructed.output_metrics.model_dump()
            == instance.output_metrics.model_dump()
        )


class TestGenerativeRequestStatsToolCalls:
    """
    Tests for tool_calls field on GenerativeRequestStats.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_tool_calls_round_trips_through_serialization(self):
        """
        tool_calls survives model_dump / model_validate.

        ## WRITTEN BY AI ##
        """
        tool_calls = [
            ToolCall(
                id="call_1",
                function=ToolCallFunction(name="fn", arguments="{}"),
            )
        ]
        tool_calls_dicts = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "fn", "arguments": "{}"},
            }
        ]
        timings = RequestTimings(resolve_start=1.0, resolve_end=2.0)
        stats = GenerativeRequestStats(
            request_id="rt-1",
            info=RequestInfo(request_id="rt-1", status="completed", timings=timings),
            input_metrics=UsageMetrics(),
            output_metrics=UsageMetrics(),
            tool_calls=tool_calls,
        )
        data = stats.model_dump()
        assert data["tool_calls"] == tool_calls_dicts

        restored = GenerativeRequestStats.model_validate(data)
        assert restored.tool_calls == tool_calls

    @pytest.mark.smoke
    def test_tool_calls_defaults_to_none(self):
        """
        tool_calls is None when not provided.

        ## WRITTEN BY AI ##
        """
        stats = GenerativeRequestStats(
            request_id="rt-2",
            info=RequestInfo(request_id="rt-2", status="completed"),
            input_metrics=UsageMetrics(),
            output_metrics=UsageMetrics(),
        )
        assert stats.tool_calls is None


class _WordTokenizer:
    """Counts whitespace-separated words, and records each encoded string."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.texts.append(text)
        tokens = list(range(len(text.split())))
        if add_special_tokens:
            return [-1, *tokens]
        return tokens


def _early_stop_stream() -> tuple[GenerationRequest, GenerationResponse, RequestInfo]:
    """
    Ten content chunks and no usage chunk, shorter than the request target.

    Mirrors the streaming server in issue #1531: the client asked for 30 output
    tokens, and the server returned 10 words with no ``usage``.
    """
    request = GenerationRequest(
        request_id="req-early-stop",
        input_metrics=UsageMetrics(text_tokens=20),
        output_metrics=UsageMetrics(text_tokens=30),
    )
    handler = ChatCompletionsRequestHandler()
    arguments = GenerationRequestArguments()
    for index in range(10):
        handler.add_streaming_line(
            f'data: {{"choices": [{{"delta": {{"content": "w{index} "}}}}]}}'
        )
    handler.add_streaming_line("data: [DONE]")
    response = handler.compile_streaming(request, arguments)
    info = RequestInfo(
        request_id=request.request_id,
        status="completed",
        timings=RequestTimings(
            request_start=0.0,
            first_token_iteration=0.02,
            last_token_iteration=0.20,
            request_end=0.30,
            resolve_end=0.30,
        ),
    )
    return request, response, info


class TestCompileStatsMissingUsage:
    """
    Output token counts when the server omits usage.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.regression
    def test_missing_usage_falls_back_to_request_target_without_tokenizer(self):
        """
        With no tokenizer, a response that omits usage keeps the request target.

        ## WRITTEN BY AI ##
        """
        request, response, info = _early_stop_stream()
        assert response.output_metrics.text_tokens is None
        assert response.text == " ".join(f"w{index}" for index in range(10)) + " "

        stats = response.compile_stats(request, info)

        assert stats.output_metrics.text_tokens == 30
        assert stats.output_tokens == 30
        assert stats.output_tokens_per_second == pytest.approx(100.0)

    @pytest.mark.regression
    def test_missing_usage_counts_streamed_text_with_tokenizer(self):
        """
        A stream with no usage counts output tokens from the response text.

        The request target is 30 and the stream is 10 words. Throughput follows
        the counted text, and prompt tokens still come from the request.

        ## WRITTEN BY AI ##
        """
        request, response, info = _early_stop_stream()
        tokenizer = _WordTokenizer()

        stats = response.compile_stats(
            request, info, tokenizer_factory=lambda: tokenizer
        )

        assert tokenizer.texts == [response.text]
        assert stats.output_metrics.text_tokens == 10
        assert stats.output_tokens == 10
        assert stats.input_metrics.text_tokens == 20
        assert stats.output_tokens_per_second == pytest.approx(10 / 0.30)
        assert stats.time_per_output_token_ms == pytest.approx(1000 * 0.20 / 10)
        assert stats.inter_token_latency_ms == pytest.approx(1000 * (0.20 - 0.02) / 9)

    @pytest.mark.regression
    def test_reported_usage_is_not_replaced_by_tokenizer(self):
        """
        Output tokens from the server are kept when usage is present.

        ## WRITTEN BY AI ##
        """
        request, response, info = _early_stop_stream()
        response.output_metrics.text_tokens = 7
        tokenizer = _WordTokenizer()

        stats = response.compile_stats(
            request, info, tokenizer_factory=lambda: tokenizer
        )

        assert tokenizer.texts == []
        assert stats.output_metrics.text_tokens == 7

    @pytest.mark.regression
    def test_request_metrics_preferred_keeps_target(self):
        """
        prefer_response False keeps the request target even when a tokenizer
        is available.

        ## WRITTEN BY AI ##
        """
        request, response, info = _early_stop_stream()
        tokenizer = _WordTokenizer()

        stats = response.compile_stats(
            request, info, prefer_response=False, tokenizer_factory=lambda: tokenizer
        )

        assert tokenizer.texts == []
        assert stats.output_metrics.text_tokens == 30

    @pytest.mark.regression
    def test_counted_tokens_omit_special_tokens(self):
        """
        A tokenizer that prepends BOS does not inflate the early-stop count.

        ## WRITTEN BY AI ##
        """
        request, response, info = _early_stop_stream()
        tokenizer = _WordTokenizer()

        stats = response.compile_stats(
            request, info, tokenizer_factory=lambda: tokenizer
        )

        assert response.text is not None
        assert len(tokenizer.encode(response.text)) == 11
        assert stats.output_metrics.text_tokens == 10
