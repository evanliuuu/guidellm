"""Intermediate benchmark reports written while a strategy is still running."""

from __future__ import annotations

import os
from pathlib import Path
from time import monotonic
from typing import Literal

from guidellm.benchmark.outputs.serialized import GenerativeBenchmarkerSerialized
from guidellm.benchmark.profiles import Profile
from guidellm.benchmark.progress import BenchmarkerProgress
from guidellm.benchmark.schemas import (
    GenerativeBenchmark,
    GenerativeBenchmarkAccumulator,
    GenerativeBenchmarksReport,
)
from guidellm.logger import logger
from guidellm.scheduler import SchedulerState, SchedulingStrategy
from guidellm.schemas.benchmark import BenchmarkOutputArgs, BenchmarkScenario
from guidellm.schemas.benchmark.outputs import (
    JSONBenchmarkOutputArgs,
    YAMLBenchmarkOutputArgs,
)
from guidellm.settings import settings

__all__ = [
    "CheckpointReportWriter",
    "GenerativeCheckpointBenchmarkerProgress",
    "resolve_checkpoint_writer",
]


class CheckpointReportWriter:
    """
    Atomically replace one checkpoint file with a serialized report.

    The bytes are written to a temporary file in the same directory and then
    moved into place, so a failed write leaves the previous checkpoint as it was.
    """

    def __init__(
        self,
        path: Path,
        format_type: Literal["json", "yaml"],
        scenario: BenchmarkScenario,
    ) -> None:
        """
        :param path: Checkpoint file to replace
        :param format_type: Serialization format for the report
        :param scenario: Scenario stored on every checkpoint report
        """
        self.path = path
        self.format_type = format_type
        self.scenario = scenario

    async def write(self, report: GenerativeBenchmarksReport) -> Path:
        """
        Replace the checkpoint file with ``report``.

        :param report: Finished strategies plus the in-progress snapshot
        :return: Path written
        :raises OSError: If the report cannot be written or replaced
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            report.save_file(temporary, type_=self.format_type)
            os.replace(temporary, self.path)  # noqa: PTH105
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return self.path


def resolve_checkpoint_writer(
    outputs: list[BenchmarkOutputArgs],
    scenario: BenchmarkScenario,
) -> CheckpointReportWriter:
    """
    Choose the checkpoint file for this run.

    The path is ``<stem>.checkpoint<suffix>`` of the first JSON or YAML output.
    A directory uses that handler's ``benchmarks.<format>`` name first. With no
    JSON or YAML output, the file is ``benchmarks.checkpoint.json`` in the
    default results directory.

    :param outputs: Output arguments configured for the run
    :param scenario: Scenario stored on every checkpoint report
    :return: Writer that replaces the checkpoint file
    """
    for args in outputs:
        if isinstance(args, JSONBenchmarkOutputArgs | YAMLBenchmarkOutputArgs):
            final = GenerativeBenchmarkerSerialized.from_args(args).output_file()
            path = final.with_name(f"{final.stem}.checkpoint{final.suffix}")
            format_type: Literal["json", "yaml"] = (
                "yaml" if path.suffix.lower() in {".yaml", ".yml"} else "json"
            )
            return CheckpointReportWriter(path, format_type, scenario)
    return CheckpointReportWriter(
        settings.default_results_dir / "benchmarks.checkpoint.json",
        "json",
        scenario,
    )


class GenerativeCheckpointBenchmarkerProgress(
    BenchmarkerProgress[GenerativeBenchmarkAccumulator, GenerativeBenchmark]
):
    """
    Write a checkpoint report on the progress update interval.

    The clock starts again at each strategy. A checkpoint contains every
    strategy that has already finished and a compile of the one still running.
    """

    def __init__(self, interval: float, writer: CheckpointReportWriter) -> None:
        """
        :param interval: Seconds to wait after a strategy starts, and between
            later checkpoints
        :param writer: Replaces the checkpoint file
        """
        super().__init__()
        self.interval = interval
        self._writer = writer
        self._scenario = writer.scenario
        self._completed: list[GenerativeBenchmark] = []
        self._checkpoint_at = 0.0

    async def on_initialize(self, profile: Profile) -> None:
        """
        :param profile: Benchmark profile for this run
        """
        self.profile = profile
        self._completed = []

    async def on_benchmark_start(self, strategy: SchedulingStrategy) -> None:
        """
        Start the interval clock for this strategy.

        :param strategy: Strategy that is about to run
        """
        self.current_strategy = strategy
        self._checkpoint_at = monotonic()

    async def on_benchmark_update(
        self,
        accumulator: GenerativeBenchmarkAccumulator,
        scheduler_state: SchedulerState,
    ) -> None:
        """
        Compile and write when the interval has elapsed.

        :param accumulator: Requests completed so far for the running strategy
        :param scheduler_state: Scheduler state for the request just applied
        """
        if monotonic() - self._checkpoint_at < self.interval:
            return
        try:
            snapshot = GenerativeBenchmark.compile(
                accumulator=accumulator,
                scheduler_state=scheduler_state,
            )
        except ValueError as err:
            logger.debug("Checkpoint skipped, no measurement window yet: {}", err)
            return
        report = GenerativeBenchmarksReport(
            config=self._scenario,
            benchmarks=[*self._completed, snapshot],
        )
        await self._writer.write(report)
        self._checkpoint_at = monotonic()

    async def on_benchmark_complete(self, benchmark: GenerativeBenchmark) -> None:
        """
        Keep a finished strategy for the checkpoints that follow it.

        :param benchmark: Compiled result for the strategy that just finished
        """
        self._completed.append(benchmark)

    async def on_finalize(self) -> None:
        """Leave the last checkpoint file in place."""
