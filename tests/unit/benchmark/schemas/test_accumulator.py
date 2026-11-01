from __future__ import annotations

import pytest

from guidellm.benchmark.schemas import BenchmarkConfig, GenerativeBenchmarkAccumulator
from guidellm.benchmark.schemas.accumulator import GenerativeRequestsAccumulator
from guidellm.scheduler import SchedulerState, SynchronousStrategy
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    GenerativeRequestStats,
    RequestInfo,
    RequestTimings,
    UsageMetrics,
)
from guidellm.schemas.base.response import OutputTokenizerFactory


def _make_stats(
    request_id: str = "req-1",
    output: str | None = "some output",
    reasoning_output: str | None = None,
    request_args: str | None = "args",
) -> GenerativeRequestStats:
    """Build a minimal GenerativeRequestStats for testing."""
    info = RequestInfo(request_id=request_id, status="completed")
    info.timings.request_start = 0.0
    info.timings.request_end = 1.0
    info.timings.resolve_end = 1.0
    return GenerativeRequestStats(
        request_id=request_id,
        request_args=request_args,
        output=output,
        reasoning_output=reasoning_output,
        info=info,
        input_metrics=UsageMetrics(text_tokens=5),
        output_metrics=UsageMetrics(text_tokens=10),
    )


class TestClearStatsData:
    """
    Tests for GenerativeRequestsAccumulator.clear_stats_data.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_clears_output_and_reasoning_output(self):
        """
        clear_stats_data clears output, reasoning_output, and tool_calls
        when clear_nonsampled_outputs is True.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator(
            clear_nonsampled_outputs=True,
            clear_nonsampled_request_args=False,
        )
        stats = _make_stats(
            output="answer",
            reasoning_output="thinking...",
            request_args="args",
        )

        acc.clear_stats_data(stats)

        assert stats.output is None
        assert stats.reasoning_output is None
        assert stats.tool_calls is None
        assert stats.request_args == "args"

    @pytest.mark.smoke
    def test_clears_request_args(self):
        """
        clear_stats_data clears request_args when
        clear_nonsampled_request_args is True.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator(
            clear_nonsampled_outputs=False,
            clear_nonsampled_request_args=True,
        )
        stats = _make_stats(
            output="answer",
            reasoning_output="thinking...",
            request_args="args",
        )

        acc.clear_stats_data(stats)

        assert stats.request_args is None
        assert stats.output == "answer"
        assert stats.reasoning_output == "thinking..."

    @pytest.mark.smoke
    def test_clears_both(self):
        """
        clear_stats_data clears all fields when both flags are True.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator(
            clear_nonsampled_outputs=True,
            clear_nonsampled_request_args=True,
        )
        stats = _make_stats(
            output="answer",
            reasoning_output="thinking...",
            request_args="args",
        )

        acc.clear_stats_data(stats)

        assert stats.request_args is None
        assert stats.output is None
        assert stats.reasoning_output is None
        assert stats.tool_calls is None

    @pytest.mark.smoke
    def test_clears_by_index(self):
        """
        clear_stats_data accepts an integer index to look up the stats
        in requests_stats.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator(
            clear_nonsampled_outputs=True,
            clear_nonsampled_request_args=True,
        )
        stats = _make_stats(reasoning_output="step 1")
        acc.requests_stats.append(stats)

        acc.clear_stats_data(0)

        assert stats.output is None
        assert stats.reasoning_output is None
        assert stats.request_args is None

    @pytest.mark.smoke
    def test_preserves_when_both_flags_false(self):
        """
        clear_stats_data preserves all fields when both flags are False.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator(
            clear_nonsampled_outputs=False,
            clear_nonsampled_request_args=False,
        )
        stats = _make_stats(
            output="answer",
            reasoning_output="thinking...",
            request_args="args",
        )

        acc.clear_stats_data(stats)

        assert stats.request_args == "args"
        assert stats.output == "answer"
        assert stats.reasoning_output == "thinking..."


def _make_timed_stats(
    request_id: str,
    request_start: float,
    request_end: float,
) -> GenerativeRequestStats:
    """Build a GenerativeRequestStats with explicit request timing bounds."""
    info = RequestInfo(request_id=request_id, status="completed")
    info.timings.request_start = request_start
    info.timings.request_end = request_end
    info.timings.resolve_end = request_end
    return GenerativeRequestStats(
        request_id=request_id,
        info=info,
        input_metrics=UsageMetrics(text_tokens=5),
        output_metrics=UsageMetrics(text_tokens=10),
    )


class TestGetWithinRange:
    """
    Tests for GenerativeRequestsAccumulator.get_within_range.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.sanity
    def test_partial_overlap_includes_events_crossing_boundary(self):
        """
        Default closed=False includes events that partially overlap the window.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator()
        acc.requests_stats = [
            _make_timed_stats("before", 5.0, 12.0),
            _make_timed_stats("inside", 11.0, 18.0),
            _make_timed_stats("after", 18.0, 24.0),
        ]

        in_range = acc.get_within_range(10.0, 20.0)

        assert [stats.request_id for stats in in_range] == [
            "before",
            "inside",
            "after",
        ]

    @pytest.mark.sanity
    def test_closed_range_requires_full_enclosure(self):
        """
        closed=True excludes boundary-crossing events and keeps only fully
        enclosed ones.

        ## WRITTEN BY AI ##
        """
        acc = GenerativeRequestsAccumulator()
        acc.requests_stats = [
            _make_timed_stats("before", 5.0, 12.0),
            _make_timed_stats("inside", 11.0, 18.0),
            _make_timed_stats("after", 18.0, 24.0),
        ]

        in_range = acc.get_within_range(10.0, 20.0, closed=True)

        assert [stats.request_id for stats in in_range] == ["inside"]


class _WordTokenizer:
    """Counts whitespace-separated words."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = list(range(len(text.split())))
        if add_special_tokens:
            return [-1, *tokens]
        return tokens


def _benchmark_accumulator(
    factory: OutputTokenizerFactory | None = None,
) -> GenerativeBenchmarkAccumulator:
    return GenerativeBenchmarkAccumulator(
        output_tokenizer_factory=factory,
        config=BenchmarkConfig(
            run_id="missing-usage",
            run_index=0,
            strategy=SynchronousStrategy(),
            constraints={},
            profile={},
            requests={},
            backend={},
            environment={},
        ),
    )


def _completed_info(request_id: str) -> RequestInfo:
    return RequestInfo(
        request_id=request_id,
        status="completed",
        timings=RequestTimings(
            queued=0.0,
            dequeued=0.01,
            resolve_start=0.02,
            request_start=0.0,
            first_token_iteration=0.02,
            last_token_iteration=0.20,
            request_end=0.30,
            resolve_end=0.30,
            finalized=0.31,
        ),
    )


class TestMissingUsageOutputTokens:
    """
    Benchmark accumulation counts output tokens when usage is omitted.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.regression
    def test_factory_counts_text_shorter_than_request_target(self):
        """
        The configured tokenizer replaces the request target for a completed
        response that has text and no output token count.

        ## WRITTEN BY AI ##
        """
        loads: list[int] = []

        def factory() -> _WordTokenizer:
            loads.append(1)
            return _WordTokenizer()

        accumulator = _benchmark_accumulator(factory)
        request = GenerationRequest(
            request_id="req-1",
            input_metrics=UsageMetrics(text_tokens=20),
            output_metrics=UsageMetrics(text_tokens=30),
        )
        response = GenerationResponse(
            request_id="req-1",
            request_args=None,
            text=" ".join(f"w{index}" for index in range(10)),
            output_metrics=UsageMetrics(),
        )

        accumulator.update_estimate(
            response, request, _completed_info("req-1"), SchedulerState()
        )

        stats = accumulator.completed.requests_stats[0]
        assert loads == [1]
        assert stats.output_metrics.text_tokens == 10
        assert stats.output_tokens == 10
        assert stats.input_metrics.text_tokens == 20

        used = GenerationResponse(
            request_id="req-2",
            request_args=None,
            text="one two three four",
            output_metrics=UsageMetrics(text_tokens=7),
        )
        used_request = GenerationRequest(
            request_id="req-2",
            output_metrics=UsageMetrics(text_tokens=30),
        )
        accumulator.update_estimate(
            used, used_request, _completed_info("req-2"), SchedulerState()
        )

        assert loads == [1]
        assert accumulator.completed.requests_stats[1].output_metrics.text_tokens == 7

    @pytest.mark.regression
    def test_tokenizer_factory_absent_from_model_dump(self):
        """
        The tokenizer factory is not part of the serialized accumulator.

        ## WRITTEN BY AI ##
        """

        def factory() -> _WordTokenizer:
            return _WordTokenizer()

        accumulator = _benchmark_accumulator(factory)
        dumped = accumulator.model_dump()
        dumped_json = accumulator.model_dump(mode="json")

        assert "output_tokenizer_factory" not in dumped
        assert "_output_tokenizer_factory" not in dumped
        assert "output_tokenizer_factory" not in dumped_json
        assert "_output_tokenizer_factory" not in dumped_json
