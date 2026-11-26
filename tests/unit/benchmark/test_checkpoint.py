"""Intermediate reports while strategies are still running."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import yaml
from click.testing import CliRunner
from pydantic import ValidationError

import guidellm.entrypoints as text_entry
from guidellm.__main__ import cli
from guidellm.benchmark import benchmarker as benchmarker_module
from guidellm.benchmark import entrypoints as entrypoints_module
from guidellm.benchmark.outputs import checkpoint as checkpoint_module
from guidellm.benchmark.outputs.checkpoint import (
    CheckpointReportWriter,
    GenerativeCheckpointBenchmarkerProgress,
    resolve_checkpoint_writer,
)
from guidellm.benchmark.schemas import (
    BenchmarkConfig,
    GenerativeBenchmark,
    GenerativeBenchmarkAccumulator,
    GenerativeBenchmarksReport,
)
from guidellm.logger import logger
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
    YAMLBenchmarkOutputArgs,
)
from guidellm.settings import settings


def _scenario(**kwargs) -> BenchmarkScenario:
    spec = {
        "backend": {"kind": "openai_http", "target": "http://localhost:9"},
        "profile": {"kind": "synchronous"},
        "data": [{"kind": "synthetic_text", "prompt_tokens": 8, "output_tokens": 4}],
        "metrics": {"kind": "generative", "warnings": []},
    }
    spec.update(kwargs.pop("spec_updates", {}))
    return BenchmarkScenario(spec=spec, **kwargs)  # type: ignore[arg-type]


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
        successful_requests=1,
        processed_requests=1,
    )
    return response, request, info, state


def _ids(benchmark: GenerativeBenchmark) -> list[str]:
    return [stats.request_id for stats in benchmark.requests.successful or []]


@pytest.mark.sanity
@pytest.mark.asyncio
async def test_run_checkpoint_keeps_finished_strategies(monkeypatch, tmp_path: Path):
    """A two-strategy run checkpoints the finished strategy plus the live one.

    ## WRITTEN BY AI ##
    """
    final_path = tmp_path / "results.json"
    clock = {"now": 0.0}
    count = {"n": 0}
    monkeypatch.setattr(checkpoint_module, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        benchmarker_module.InfoMixin, "extract_from_obj", lambda _obj: {}
    )
    monkeypatch.setattr(benchmarker_module, "prefetch_requests", lambda *_a, **_k: [])

    async def schedule(**_kwargs):
        clock["now"] += 20.0
        index = count["n"]
        count["n"] += 1
        yield _completed(index)

    monkeypatch.setattr(
        benchmarker_module, "Scheduler", Mock(return_value=Mock(run=schedule))
    )
    monkeypatch.setattr(
        entrypoints_module,
        "resolve_backend",
        AsyncMock(return_value=(MagicMock(), "model")),
    )
    monkeypatch.setattr(entrypoints_module, "resolve_tokenizer", AsyncMock())
    monkeypatch.setattr(
        entrypoints_module, "create_data_loader", AsyncMock(return_value=[])
    )

    scenario = _scenario(
        checkpoint_interval=15.0,
        spec_updates={
            "profile": {"kind": "concurrent", "streams": [1, 2]},
            "outputs": [{"kind": "json", "path": str(final_path)}],
        },
    )
    report, _outputs = await entrypoints_module.benchmark_generative_text(
        args=scenario,
        progress=False,
        console=None,
    )

    checkpoint = tmp_path / "results.checkpoint.json"
    loaded = GenerativeBenchmarksReport.load_file(checkpoint)
    assert [benchmark.config.strategy.streams for benchmark in loaded.benchmarks] == [
        1,
        2,
    ]
    assert [_ids(benchmark) for benchmark in loaded.benchmarks] == [
        ["req-0"],
        ["req-1"],
    ]
    assert len(report.benchmarks) == 2
    assert checkpoint != final_path
    assert final_path.exists()

    count["n"] = 0
    clock["now"] = 0.0
    absent = tmp_path / "absent.json"
    quiet = _scenario(
        spec_updates={
            "profile": {"kind": "concurrent", "streams": [1, 2]},
            "outputs": [{"kind": "json", "path": str(absent)}],
        }
    )
    await entrypoints_module.benchmark_generative_text(
        args=quiet,
        progress=False,
        console=None,
    )
    assert not (tmp_path / "absent.checkpoint.json").exists()


@pytest.mark.regression
def test_failed_checkpoint_write_keeps_the_previous_file(monkeypatch, tmp_path: Path):
    """A write that fails before replace leaves the last checkpoint unchanged.

    ## WRITTEN BY AI ##
    """
    path = tmp_path / "benchmarks.checkpoint.json"
    path.write_text("previous", encoding="utf-8")
    writer = CheckpointReportWriter(path, "json", _scenario())

    def fail(self, destination, type_):
        Path(destination).write_text("partial", encoding="utf-8")
        raise OSError(f"disk full writing {type_}")

    monkeypatch.setattr(GenerativeBenchmarksReport, "save_file", fail)
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(writer.write(GenerativeBenchmarksReport(config=writer.scenario)))

    assert path.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.glob(".benchmarks.checkpoint.json.*.tmp")) == []


@pytest.mark.regression
def test_checkpoint_path_uses_serialized_output_or_the_default_dir(tmp_path: Path):
    """JSON and YAML paths gain a checkpoint suffix. Anything else uses the default.

    ## WRITTEN BY AI ##
    """
    scenario = _scenario()
    json_path = tmp_path / "run" / "results.json"
    writer = resolve_checkpoint_writer(
        [
            CSVBenchmarkOutputArgs(path=tmp_path / "run" / "results.csv"),
            JSONBenchmarkOutputArgs(path=json_path),
        ],
        scenario,
    )
    assert writer.path == tmp_path / "run" / "results.checkpoint.json"

    directory = tmp_path / "reports"
    directory.mkdir()
    yaml_writer = resolve_checkpoint_writer(
        [YAMLBenchmarkOutputArgs(path=directory)],
        scenario,
    )
    assert yaml_writer.path == directory / "benchmarks.checkpoint.yaml"

    original = settings.default_results_dir
    settings.default_results_dir = tmp_path / "default"
    try:
        fallback = resolve_checkpoint_writer(
            [CSVBenchmarkOutputArgs(path=tmp_path / "only.csv")],
            scenario,
        )
    finally:
        settings.default_results_dir = original
    assert fallback.path == tmp_path / "default" / "benchmarks.checkpoint.json"


@pytest.mark.regression
@pytest.mark.asyncio
async def test_checkpoint_clock_resets_when_the_next_strategy_starts(
    monkeypatch, tmp_path: Path
):
    """A new strategy waits out its own interval before the next checkpoint.

    ## WRITTEN BY AI ##
    """
    clock = {"now": 0.0}
    monkeypatch.setattr(checkpoint_module, "monotonic", lambda: clock["now"])
    written: list[int] = []
    writer = CheckpointReportWriter(
        tmp_path / "benchmarks.checkpoint.json", "json", _scenario()
    )

    async def record(report: GenerativeBenchmarksReport) -> Path:
        written.append(len(report.benchmarks))
        return writer.path

    monkeypatch.setattr(writer, "write", record)
    tracker = GenerativeCheckpointBenchmarkerProgress(15.0, writer)
    config = BenchmarkConfig(
        run_id="checkpoint",
        run_index=0,
        strategy=SynchronousStrategy(),
        constraints={},
        profile={},
        requests={},
        backend={},
        environment={},
        warnings=[],
    )
    accumulator = GenerativeBenchmarkAccumulator(config=config)
    response, request, info, state = _completed(0)
    accumulator.update_estimate(response, request, info, state)
    finished = GenerativeBenchmark.compile(
        accumulator=accumulator, scheduler_state=state
    )

    await tracker.on_benchmark_start(SynchronousStrategy())
    clock["now"] = 20.0
    await tracker.on_benchmark_update(accumulator, state)
    await tracker.on_benchmark_complete(finished)
    clock["now"] = 100.0
    await tracker.on_benchmark_start(SynchronousStrategy())
    clock["now"] = 110.0
    await tracker.on_benchmark_update(accumulator, state)
    clock["now"] = 120.0
    await tracker.on_benchmark_update(accumulator, state)

    assert written == [1, 2]


@pytest.mark.regression
@pytest.mark.asyncio
async def test_missing_measurement_window_is_logged_at_debug(
    monkeypatch, tmp_path: Path
):
    """A compile with no measurement window is logged and does not write.

    ## WRITTEN BY AI ##
    """
    clock = {"now": 0.0}
    monkeypatch.setattr(checkpoint_module, "monotonic", lambda: clock["now"])
    records: list[str] = []
    sink = logger.add(lambda message: records.append(str(message)), level="DEBUG")
    writer = CheckpointReportWriter(
        tmp_path / "benchmarks.checkpoint.json", "json", _scenario()
    )
    called = {"write": False}

    async def record(_report):
        called["write"] = True
        return writer.path

    monkeypatch.setattr(writer, "write", record)
    tracker = GenerativeCheckpointBenchmarkerProgress(15.0, writer)
    accumulator = GenerativeBenchmarkAccumulator(
        config=BenchmarkConfig(
            run_id="checkpoint",
            run_index=0,
            strategy=SynchronousStrategy(),
            constraints={},
            profile={},
            requests={},
            backend={},
            environment={},
            warnings=[],
        )
    )
    try:
        await tracker.on_benchmark_start(SynchronousStrategy())
        clock["now"] = 20.0
        await tracker.on_benchmark_update(accumulator, SchedulerState())
    finally:
        logger.remove(sink)

    assert called["write"] is False
    assert any("no measurement window yet" in record for record in records)


@pytest.mark.regression
@pytest.mark.parametrize("value", [0, -1])
def test_checkpoint_interval_rejects_non_positive(value: int):
    """Zero and negative intervals are rejected by the scenario field.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValidationError):
        _scenario(checkpoint_interval=value)


@pytest.mark.regression
def test_run_and_config_pass_checkpoint_interval(tmp_path: Path, monkeypatch):
    """The run and config commands forward the flag and leave a file value alone.

    ## WRITTEN BY AI ##
    """
    seen: list[BenchmarkScenario] = []

    async def capture(args, **_kwargs):
        seen.append(args)

    monkeypatch.setattr(text_entry, "benchmark_generative_text", capture)
    runner = CliRunner()
    flagged = runner.invoke(
        cli,
        [
            "run",
            "--backend",
            "kind=openai_http,target=http://localhost:8000",
            "--profile",
            "kind=synchronous",
            "--data",
            "kind=synthetic_text,prompt_tokens=8",
            "--checkpoint-interval",
            "600",
            "--disable-console",
        ],
    )
    assert flagged.exit_code == 0, flagged.output
    assert seen[-1].checkpoint_interval == 600.0

    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "checkpoint_interval": 30,
                "spec": {
                    "backend": {
                        "kind": "openai_http",
                        "target": "http://localhost:8000",
                    },
                    "profile": {"kind": "synchronous"},
                    "data": [{"kind": "synthetic_text", "prompt_tokens": 8}],
                },
            }
        ),
        encoding="utf-8",
    )
    kept = runner.invoke(
        cli,
        ["run", "--config", str(source), "--disable-console"],
    )
    assert kept.exit_code == 0, kept.output
    assert seen[-1].checkpoint_interval == 30.0

    written = tmp_path / "written.yaml"
    configured = runner.invoke(
        cli,
        [
            "config",
            str(written),
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
    assert configured.exit_code == 0, configured.output
    assert BenchmarkScenario.create(scenario=written).checkpoint_interval == 600.0
