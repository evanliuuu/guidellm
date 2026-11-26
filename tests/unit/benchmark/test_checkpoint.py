"""Intermediate reports while one strategy is still running."""

from pathlib import Path
from typing import NamedTuple
from unittest.mock import AsyncMock, Mock

import click
import pytest
from click.testing import CliRunner

from guidellm.__main__ import cli
from guidellm.benchmark import benchmarker as benchmarker_module
from guidellm.benchmark.entrypoints import (
    checkpoint_report_path,
    write_checkpoint_report,
)
from guidellm.benchmark.schemas import (
    GenerativeBenchmark,
    GenerativeBenchmarkAccumulator,
    GenerativeBenchmarksReport,
)
from guidellm.scheduler import SchedulerState, SynchronousStrategy
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
    RequestTimings,
    UsageMetrics,
)
from guidellm.schemas.benchmark import BenchmarkScenario
from guidellm.schemas.benchmark.outputs import (
    CSVBenchmarkOutputArgs,
    JSONBenchmarkOutputArgs,
)
from guidellm.schemas.benchmark.transient import TransientPhaseConfig


def _scenario(path: Path) -> BenchmarkScenario:
    return BenchmarkScenario(
        spec={  # type: ignore[arg-type]
            "backend": {"kind": "openai_http", "target": "http://localhost:9"},
            "profile": {"kind": "synchronous"},
            "data": [
                {"kind": "synthetic_text", "prompt_tokens": 8, "output_tokens": 4}
            ],
            "outputs": [{"kind": "json", "path": str(path)}],
        }
    )


def _completed(index: int):
    request_id = f"req-{index}"
    start = 1000.0 + index
    timings = RequestTimings(
        queued=start,
        dequeued=start + 0.01,
        resolve_start=start + 0.02,
        request_start=start + 0.03,
        first_token_iteration=start + 0.10,
        last_token_iteration=start + 0.30,
        request_end=start + 0.31,
        resolve_end=start + 0.32,
        finalized=start + 0.33,
        token_iterations=4,
    )
    info = RequestInfo(request_id=request_id, status="completed", timings=timings)
    request = GenerationRequest(
        request_id=request_id,
        input_metrics=UsageMetrics(text_tokens=8),
        output_metrics=UsageMetrics(text_tokens=4),
    )
    response = GenerationResponse(
        request_id=request_id,
        request_args="",
        input_metrics=UsageMetrics(text_tokens=8),
        output_metrics=UsageMetrics(text_tokens=4),
    )
    state = SchedulerState(
        start_time=1000.0,
        start_requests_time=1000.0,
        successful_requests=index + 1,
        processed_requests=index + 1,
    )
    return response, request, info, state


def _in_progress(request_id: str):
    timings = RequestTimings(resolve_start=1000.2, request_start=1000.2)
    info = RequestInfo(request_id=request_id, status="in_progress", timings=timings)
    request = GenerationRequest(request_id=request_id)
    response = GenerationResponse(request_id=request_id, request_args="")
    state = SchedulerState(start_time=1000.0, start_requests_time=1000.0)
    return response, request, info, state


class _View(NamedTuple):
    ids: list[str]
    completed: int
    latency_mean: float
    latency_p50: float
    ttft_mean: float
    itl_mean: float
    throughput_mean: float


def _observable(benchmark: GenerativeBenchmark) -> _View:
    latency = benchmark.metrics.request_latency.successful
    ttft_mean = benchmark.metrics.time_to_first_token_ms.successful.mean
    itl_mean = benchmark.metrics.inter_token_latency_ms.successful.mean
    throughput_mean = benchmark.metrics.requests_per_second.successful.mean
    assert latency.mean is not None
    assert ttft_mean is not None
    assert itl_mean is not None
    assert throughput_mean is not None
    return _View(
        ids=[stats.request_id for stats in benchmark.requests.successful or []],
        completed=latency.count,
        latency_mean=latency.mean,
        latency_p50=latency.percentiles.p50,
        ttft_mean=ttft_mean,
        itl_mean=itl_mean,
        throughput_mean=throughput_mean,
    )


async def _run(monkeypatch, *, interval: float | None, on_checkpoint):
    clock = {"now": 1000.0}
    monkeypatch.setattr(benchmarker_module, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        benchmarker_module.InfoMixin, "extract_from_obj", lambda _obj: {}
    )
    monkeypatch.setattr(benchmarker_module, "prefetch_requests", lambda *_a, **_k: [])

    strategy = SynchronousStrategy()

    def strategies():
        yield strategy, {}

    profile = Mock(completed_strategies=[])
    profile.strategies_generator.side_effect = strategies

    updates = [
        ("completed", 0, 10.0),
        ("completed", 1, 10.0),
        ("in_progress", "still-running", 16.0),
        ("completed", 2, 14.0),
        ("completed", 3, 16.0),
    ]

    async def schedule(**_kwargs):
        for kind, index, step in updates:
            clock["now"] += step
            if kind == "completed":
                yield _completed(index)
            else:
                yield _in_progress(str(index))

    monkeypatch.setattr(
        benchmarker_module, "Scheduler", Mock(return_value=Mock(run=schedule))
    )
    progress = Mock()
    progress.on_initialize = AsyncMock()
    progress.on_benchmark_start = AsyncMock()
    progress.on_benchmark_update = AsyncMock()
    progress.on_benchmark_compile = AsyncMock()
    progress.on_benchmark_complete = AsyncMock()
    progress.on_finalize = AsyncMock()

    return [
        benchmark
        async for benchmark in benchmarker_module.Benchmarker().run(
            accumulator_class=GenerativeBenchmarkAccumulator,
            benchmark_class=GenerativeBenchmark,
            requests=[],
            backend=Mock(),
            profile=profile,
            environment=Mock(),
            warmup=TransientPhaseConfig(),
            cooldown=TransientPhaseConfig(),
            progress=progress,
            warnings=[],
            checkpoint_interval=interval,
            on_checkpoint=on_checkpoint,
        )
    ]


@pytest.mark.sanity
@pytest.mark.asyncio
async def test_checkpoints_snapshot_completed_requests_without_resetting(
    monkeypatch, tmp_path: Path
):
    """Emit interval snapshots of finished requests and leave the final run intact.

    ## WRITTEN BY AI ##
    """
    final_path = tmp_path / "benchmarks.json"
    final_path.write_text("final-report", encoding="utf-8")
    scenario = _scenario(final_path)
    outputs = scenario.spec.outputs
    snapshots: list[_View] = []

    async def on_checkpoint(benchmark: GenerativeBenchmark) -> None:
        report = GenerativeBenchmarksReport(config=scenario, benchmarks=[benchmark])
        written = await write_checkpoint_report(report, outputs)
        loaded = GenerativeBenchmarksReport.load_file(written)
        assert written != final_path
        assert written.name == "benchmarks.checkpoint.json"
        assert len(loaded.benchmarks) == 1
        snapshots.append(_observable(loaded.benchmarks[0]))

    results = await _run(monkeypatch, interval=15.0, on_checkpoint=on_checkpoint)
    plain = await _run(monkeypatch, interval=None, on_checkpoint=None)

    assert [item.ids for item in snapshots] == [
        ["req-0", "req-1"],
        ["req-0", "req-1"],
        ["req-0", "req-1", "req-2", "req-3"],
    ]
    assert all(item.completed == len(item.ids) for item in snapshots)
    assert all("still-running" not in item.ids for item in snapshots)
    for item in snapshots:
        assert item.ttft_mean == pytest.approx(70.0)
        assert item.itl_mean == pytest.approx(200.0 / 3.0)
        assert item.latency_mean == pytest.approx(0.28)
        assert item.latency_p50 == pytest.approx(0.28)
        assert item.throughput_mean > 0

    assert len(results) == len(plain) == 1
    finished = _observable(results[0])
    assert finished == _observable(plain[0])
    assert finished.ids == ["req-0", "req-1", "req-2", "req-3"]
    assert final_path.read_text(encoding="utf-8") == "final-report"


@pytest.mark.regression
@pytest.mark.asyncio
async def test_checkpoint_interval_defaults_off(monkeypatch):
    """No interval means no checkpoint file and the run still completes.

    ## WRITTEN BY AI ##
    """
    called = False

    async def on_checkpoint(_benchmark: GenerativeBenchmark) -> None:
        nonlocal called
        called = True

    results = await _run(monkeypatch, interval=None, on_checkpoint=on_checkpoint)
    assert results[0].requests.successful is not None
    assert len(results[0].requests.successful) == 4
    assert called is False
    assert BenchmarkScenario.model_fields["checkpoint_interval"].default is None


@pytest.mark.regression
def test_checkpoint_file_follows_json_output_and_not_the_final_path(tmp_path: Path):
    """The checkpoint is a sibling of the serialized report, never that report.

    ## WRITTEN BY AI ##
    """
    json_path = tmp_path / "run" / "results.json"
    csv_path = tmp_path / "run" / "results.csv"
    path = checkpoint_report_path(
        [
            CSVBenchmarkOutputArgs(path=csv_path),
            JSONBenchmarkOutputArgs(path=json_path),
        ]
    )
    assert path == tmp_path / "run" / "results.checkpoint.json"
    assert path != json_path
    assert path != csv_path


@pytest.mark.regression
def test_run_checkpoint_interval_round_trips_through_config(tmp_path: Path):
    """The flag is shared by run and config and stays unset unless given.

    ## WRITTEN BY AI ##
    """
    assert "--checkpoint-interval" in {
        option
        for parameter in cli.commands["run"].params
        if isinstance(parameter, click.Option)
        for option in parameter.opts
    }
    path = tmp_path / "scenario.yaml"
    result = CliRunner().invoke(
        cli,
        [
            "config",
            str(path),
            "--backend",
            "kind=openai_http,target=http://localhost:8000",
            "--profile",
            "kind=synchronous",
            "--data",
            "kind=synthetic_text,prompt_tokens=8",
            "--checkpoint-interval",
            "600",
        ],
    )
    assert result.exit_code == 0, result.output
    loaded = BenchmarkScenario.create(scenario=path)
    assert loaded.checkpoint_interval == 600.0
