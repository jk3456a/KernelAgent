# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optimization Manager for parallel kernel optimization.

This module provides the OptimizationManager class that orchestrates
parallel kernel optimization using pluggable search strategies:
- beam_search: Maintain top-N kernels, explore M bottlenecks each
- greedy: Simple single-best optimization

Example:
    >>> manager = OptimizationManager(
    ...     strategy="beam_search",
    ...     num_workers=4,
    ...     strategy_config={"num_top_kernels": 2, "num_bottlenecks": 2},
    ... )
    >>> result = manager.run_optimization(
    ...     initial_kernel=kernel_code,
    ...     problem_file=Path("problem.py"),
    ...     test_code=test_file.read_text(),
    ...     max_rounds=20,
    ... )
"""

import logging
import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Any

from triton_kernel_agent.opt_worker_component.benchmarking.performance_metrics import (
    format_performance_summary,
)
from triton_kernel_agent.opt_worker_component.searching.history.json_db import (
    JSONProgramDatabase,
)
from triton_kernel_agent.opt_worker_component.searching.history.models import (
    ProgramEntry,
    ProgramMetrics,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.strategy import (
    SearchStrategy,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.beam_search import (
    BeamSearchStrategy,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.greedy import (
    GreedyStrategy,
)
from utils.config_injectable import config_injectable
from utils.progress import get_progress, progress_stage

# Manager-level component keys resolved by the registry
_MANAGER_LEVEL_KEYS = {"verifier", "benchmarker", "worker_runner"}


@config_injectable
class OptimizationManager:
    """Manages parallel kernel optimization with pluggable strategies.

    Supports:
    - beam_search: Current default (top-N kernels × M bottlenecks)
    - greedy: Simple single-best optimization

    Platform-specific behaviour (verification, benchmarking, worker
    orchestration) is delegated to injectable components that implement
    :class:`KernelVerifier`, :class:`KernelBenchmarker`, and
    :class:`WorkerRunner`.  When these are not supplied the default
    NVIDIA / CUDA implementations are used.
    """

    def __init__(
        self,
        strategy: str = "beam_search",
        num_workers: int = 4,
        max_rounds: int = 10,
        log_dir: Path | str | None = None,
        database_path: Path | str | None = None,
        strategy_config: dict[str, Any] | None = None,
        openai_model: str = "claude-opus-4.5",
        high_reasoning_effort: bool = True,
        bottleneck_override: str | None = None,
        platform: dict[str, str] | str | None = None,
        **worker_kwargs: Any,
    ):
        """Initialize the optimization manager.

        Args:
            strategy: Search strategy name ("beam_search" or "greedy")
            num_workers: Number of parallel workers
            max_rounds: Maximum optimization rounds
            log_dir: Directory for logs and artifacts
            database_path: Path for program database JSON file
            strategy_config: Strategy-specific configuration
            openai_model: Model name for LLM optimization
            high_reasoning_effort: Whether to use high reasoning effort
            bottleneck_override: Pre-computed bottleneck category to skip LLM analysis
            platform: Platform component config.  Can be:
                - ``None`` — use ``"nvidia"`` for all components (default)
                - a string like ``"nvidia"`` — shorthand for all components
                - a dict like ``{"verifier": "nvidia", ...}`` — per-component
            **worker_kwargs: Additional kwargs passed to OptimizationWorker
        """
        self.max_rounds = max_rounds
        self.strategy_name = strategy
        self.log_dir = (
            Path(log_dir) if log_dir else Path(tempfile.mkdtemp(prefix="opt_"))
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort
        self.bottleneck_override = bottleneck_override
        self.worker_kwargs = worker_kwargs

        # Store template overrides (also stays in worker_kwargs for forwarding)
        self.templates_config = worker_kwargs.get("templates")

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize database
        db_path = (
            Path(database_path)
            if database_path
            else self.log_dir / "program_database.json"
        )
        self.database = JSONProgramDatabase(db_path)

        # Initialize strategy
        self.strategy = self._create_strategy(
            strategy, strategy_config or {}, num_workers
        )

        # Validate worker count
        if num_workers != self.strategy.num_workers_needed:
            raise ValueError(
                f"Strategy '{strategy}' requires {self.strategy.num_workers_needed} "
                f"workers, got {num_workers}. Adjust num_workers or strategy_config."
            )

        self.num_workers = num_workers
        self.benchmark_lock = mp.Lock()
        # Semaphore to serialize NCU profiling - NCU requires exclusive GPU access
        # and has high memory overhead, so only one worker should profile at a time
        self.profiling_semaphore = mp.Semaphore(1)

        # Shared history across beam search iterations
        self.shared_history: list[
            dict
        ] = []  # List of serialized OptimizationAttempt dicts
        self.shared_reflexions: list[dict] = []  # List of serialized Reflexion dicts
        self.history_size: int = 10  # Max history entries to pass to workers

        # ── Platform components (resolved from registry) ─────────
        self._resolve_platform(platform)

        self.logger.info(
            f"OptimizationManager initialized: strategy={strategy}, workers={num_workers}"
        )

    # ------------------------------------------------------------------
    # Platform resolution
    # ------------------------------------------------------------------

    def _resolve_platform(self, platform: dict[str, str] | str | None) -> None:
        """Resolve platform components from the :mod:`platform.registry`.

        Manager-level components (``verifier``, ``benchmarker``,
        ``worker_runner``) are instantiated and stored on *self*.
        Worker-level component names are forwarded to worker processes
        via ``self.worker_kwargs["platform_config"]`` so each worker
        can resolve its own instances from the registry.
        """
        from triton_kernel_agent.platform.registry import registry

        # Expand shorthand → full per-component dict
        if platform is None or isinstance(platform, str):
            impl = platform or "nvidia"
            config = {k: impl for k in registry.list_components()}
        else:
            config = dict(platform)

        # Split manager vs worker keys
        mgr_config = {k: v for k, v in config.items() if k in _MANAGER_LEVEL_KEYS}
        worker_config = {
            k: v for k, v in config.items() if k not in _MANAGER_LEVEL_KEYS
        }

        # Resolve manager-level components (shared kwargs bag is
        # filtered per-factory by the registry)
        components = registry.create_from_config(
            mgr_config,
            log_dir=self.log_dir,
            logger=self.logger,
            benchmark_lock=self.benchmark_lock,
            profiling_semaphore=self.profiling_semaphore,
            openai_model=self.openai_model,
            high_reasoning_effort=self.high_reasoning_effort,
            bottleneck_override=self.bottleneck_override,
            worker_kwargs=self.worker_kwargs,
        )
        self.verifier = components["verifier"]
        self.benchmarker = components["benchmarker"]
        self.worker_runner = components["worker_runner"]

        # Propagate worker-level config (string names) to worker
        # processes — each worker resolves its own instances via the
        # registry so there are no pickling issues.
        if worker_config:
            self.worker_kwargs["platform_config"] = worker_config

    # ------------------------------------------------------------------
    # Logging / strategy helpers (unchanged)
    # ------------------------------------------------------------------

    def _setup_logging(self) -> logging.Logger:
        """Setup manager logging."""
        logger = logging.getLogger("OptimizationManager")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler(self.log_dir / "manager.log")
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(handler)

            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(console)

        return logger

    def _create_strategy(
        self, name: str, config: dict[str, Any], num_workers: int
    ) -> SearchStrategy:
        """Create the search strategy.

        Args:
            name: Strategy name
            config: Strategy-specific configuration
            num_workers: Number of workers

        Returns:
            Configured SearchStrategy instance

        Raises:
            ValueError: If strategy name is unknown
        """
        if name == "beam_search":
            return BeamSearchStrategy(
                num_top_kernels=config.get("num_top_kernels", 2),
                num_bottlenecks=config.get("num_bottlenecks", 2),
                database=self.database,
                logger=self.logger,
            )
        elif name == "greedy":
            return GreedyStrategy(
                database=self.database,
                max_no_improvement=config.get("max_no_improvement", 5),
                logger=self.logger,
            )
        else:
            raise ValueError(f"Unknown strategy: {name}. Use 'beam_search' or 'greedy'")

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------

    @progress_stage(
        "agent2.optimize",
        source="agent2.manager",
        message="running kernel optimization",
        result_ok=lambda result: bool(result and result.get("success")),
    )
    def run_optimization(
        self,
        initial_kernel: str,
        problem_file: Path | str,
        test_code: str | list[str],
        max_rounds: int | None = None,
        resume_from: Path | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run optimization with the configured strategy.

        Args:
            initial_kernel: Starting kernel code
            problem_file: Path to problem.py defining Model and get_inputs()
            test_code: Test code for correctness verification. Can be a single
                string or a list.
            max_rounds: Override max_rounds (optional)
            **kwargs: Additional kwargs (reserved for future use)

        Returns:
            Dict with:
                - success: bool
                - kernel_code: str | None
                - best_time_ms: float
                - total_rounds: int
                - top_kernels: list[dict]
        """
        max_rounds = max_rounds or self.max_rounds
        strategy_name = self.strategy_name
        problem_file = Path(problem_file)
        progress = get_progress()

        # Normalize test_code to list
        if isinstance(test_code, str):
            test_code = [test_code]

        self.logger.info("=" * 80)
        self.logger.info("STARTING OPTIMIZATION")
        self.logger.info("=" * 80)
        progress.emit(
            "agent2.manager_start",
            source="agent2.manager",
            message="optimization manager started",
            strategy=strategy_name,
            max_rounds=max_rounds,
            log_dir=str(self.log_dir),
        )

        # Initialize strategy with starting kernel
        initial_entry = ProgramEntry(
            program_id="initial",
            kernel_code=initial_kernel,
            metrics=ProgramMetrics(time_ms=float("inf")),
            problem_id=str(problem_file),
        )
        self.strategy.initialize(initial_entry)

        # Resume: continue a prior run instead of starting fresh. We load the
        # prior program DB, seed the strategy with its best kernel (keeping its
        # measured time so the revert logic works), and continue the round
        # numbering from where the prior run stopped.
        resume_info = self._load_resume_state(resume_from) if resume_from else None
        start_round = 1
        if resume_info is not None:
            initial_kernel = resume_info["kernel_code"]
            self.strategy.initialize(resume_info["best_program"])
            for prog in resume_info["programs"]:
                self.database.add_program(prog)
            start_round = resume_info["next_round"]
            self.logger.info(
                f"RESUMED from {resume_from}: best={resume_info['best_ms']:.4f}ms, "
                f"continuing at round {start_round}"
            )

        # Verify initial kernel correctness before investing in benchmarks/optimization
        progress.emit(
            "agent2.initial_verify",
            source="agent2.manager",
            message="verifying initial kernel",
            strategy=strategy_name,
        )
        initial_ok = self._verify_initial_kernel(
            initial_kernel, problem_file, test_code
        )
        if not initial_ok:
            return {
                "success": False,
                "kernel_code": None,
                "best_time_ms": float("inf"),
                "total_rounds": 0,
                "top_kernels": [],
                "error": "Initial kernel failed correctness verification",
            }

        # Benchmark the initial kernel first. Under remote execution this writes
        # initial_kernel.py to the artifacts dir, which the remote PyTorch
        # baseline reuses to drive kernel_subprocess.py --baseline (it needs a
        # kernel alongside the problem). Order matters: baseline must come after.
        progress.emit(
            "agent2.initial_benchmark",
            source="agent2.manager",
            message="benchmarking initial kernel",
            strategy=strategy_name,
        )
        initial_kernel_time = self._benchmark_initial_kernel(
            initial_kernel, problem_file
        )

        # Benchmark PyTorch baseline once (before spawning workers)
        progress.emit(
            "agent2.pytorch_baseline",
            source="agent2.manager",
            message="benchmarking PyTorch baseline",
            strategy=strategy_name,
        )
        pytorch_baseline = self._benchmark_pytorch_baseline(problem_file)
        initial_benchmark_result = getattr(
            self.benchmarker, "last_kernel_result", None
        )
        pytorch_benchmark_result = getattr(
            self.benchmarker, "last_reference_result", None
        )

        # Benchmark torch.compile baseline
        progress.emit(
            "agent2.pytorch_compile_baseline",
            source="agent2.manager",
            message="benchmarking torch.compile baseline",
            strategy=strategy_name,
        )
        pytorch_compile_time = self._benchmark_pytorch_compile(problem_file)

        # Trajectory log spanning the whole optimization run (one record per
        # baseline + round), written at the MANAGER level so it follows the
        # real round loop below — read by the optimization dashboard.
        from triton_kernel_agent.opt_worker_component.searching.trajectory import (
            TrajectoryWriter,
        )

        trajectory = TrajectoryWriter(self.log_dir / "trajectory.jsonl")
        trajectory.record_baseline(
            time_ms=initial_kernel_time,
            pytorch_ms=pytorch_baseline,
            sol_pct=0.0,
            bottleneck="unknown",
            kernel_performance=(
                initial_benchmark_result.get("performance")
                if isinstance(initial_benchmark_result, dict)
                else None
            ),
            pytorch_performance=(
                pytorch_benchmark_result.get("performance")
                if isinstance(pytorch_benchmark_result, dict)
                else None
            ),
        )
        if resume_info is not None:
            trajectory.record_resume(
                resumed_from=str(resume_from),
                from_round=resume_info["next_round"] - 1,
                best_ms=resume_info["best_ms"],
            )

        # Round loop — continue numbering from start_round when resuming.
        round_num = start_round - 1
        for round_num in range(start_round, start_round + max_rounds):
            self.logger.info("")
            self.logger.info(f"{'=' * 20} ROUND {round_num} {'=' * 20}")
            progress.emit(
                "agent2.round",
                source="agent2.manager",
                message=f"round {round_num} started",
                strategy=strategy_name,
                round=round_num,
                max_rounds=start_round + max_rounds - 1,
            )

            # 1. Get candidates from strategy
            candidates = self.strategy.select_candidates(round_num)
            if not candidates:
                self.logger.warning("No candidates to explore, terminating")
                break

            # 2. Spawn workers
            results = self._run_workers(
                candidates,
                round_num,
                problem_file,
                test_code,
                pytorch_baseline,
            )

            # 3. Update strategy with results
            self.strategy.update_with_results(results, round_num)

            # Log per-round winner summary
            successful = [r for r in results if r.get("success")]
            if successful:
                best = min(successful, key=lambda r: r.get("time_ms", float("inf")))
                self.logger.info(
                    f"Round {round_num} best: worker {best['worker_id']} at {best['time_ms']:.4f} ms"
                )
                progress.emit(
                    "agent2.round",
                    source="agent2.manager",
                    status="completed",
                    message=(
                        f"round {round_num} best worker {best['worker_id']} "
                        f"at {best['time_ms']:.4f} ms"
                    ),
                    strategy=strategy_name,
                    round=round_num,
                    best_worker=best["worker_id"],
                    best_time_ms=best["time_ms"],
                )
            else:
                best = None
                self.logger.info(f"Round {round_num}: no successful workers")
                progress.emit(
                    "agent2.round",
                    source="agent2.manager",
                    status="completed",
                    message=f"round {round_num}: no successful workers",
                    strategy=strategy_name,
                    round=round_num,
                )

            # Append this round to the trajectory. ``attempt`` (when present)
            # carries the SOL / bottleneck / config detail the worker measured.
            self._record_round_trajectory(
                trajectory, round_num, best, initial_kernel_time
            )

            # 4. Check termination
            if self.strategy.should_terminate(
                round_num, start_round + max_rounds - 1
            ):
                self.logger.info("Strategy signaled termination")
                break

        # Return best result
        best = self.strategy.get_best_program()

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("OPTIMIZATION COMPLETE")
        self.logger.info("=" * 80)

        best_performance = None
        if best:
            best_performance = {
                "achieved_tflops": best.metrics.achieved_tflops,
                "mfu_pct": best.metrics.mfu_pct,
                "roofline_attainable_tflops": (
                    best.metrics.roofline_attainable_tflops
                ),
                "roofline_utilization_pct": (
                    best.metrics.roofline_utilization_pct
                ),
            }
            self.logger.info(f"Best time: {best.metrics.time_ms:.4f}ms")
            if best.metrics.achieved_tflops is not None:
                self.logger.info(
                    "Best compute: "
                    + format_performance_summary(best_performance)
                )
            if initial_kernel_time != float("inf") and best.metrics.time_ms > 0:
                speedup = initial_kernel_time / best.metrics.time_ms
                self.logger.info(f"Speedup vs initial kernel: {speedup:.2f}x")
            if pytorch_baseline != float("inf") and best.metrics.time_ms > 0:
                speedup_pt = pytorch_baseline / best.metrics.time_ms
                self.logger.info(f"Speedup vs PyTorch eager: {speedup_pt:.2f}x")

        return {
            "success": best is not None and best.metrics.time_ms != float("inf"),
            "kernel_code": best.kernel_code if best else None,
            "best_time_ms": best.metrics.time_ms if best else float("inf"),
            "total_rounds": round_num,
            "pytorch_baseline_ms": pytorch_baseline,
            "pytorch_compile_ms": pytorch_compile_time,
            "initial_kernel_time_ms": initial_kernel_time,
            "best_performance": best_performance,
            "initial_kernel_performance": (
                initial_benchmark_result.get("performance")
                if isinstance(initial_benchmark_result, dict)
                else None
            ),
            "pytorch_baseline_performance": (
                pytorch_benchmark_result.get("performance")
                if isinstance(pytorch_benchmark_result, dict)
                else None
            ),
            "pytorch_baseline_backend": (
                pytorch_benchmark_result.get("backend")
                if isinstance(pytorch_benchmark_result, dict)
                else None
            ),
            "top_kernels": [
                {
                    "kernel_code": p.kernel_code,
                    "time_ms": p.metrics.time_ms,
                    "performance": {
                        "achieved_tflops": p.metrics.achieved_tflops,
                        "mfu_pct": p.metrics.mfu_pct,
                        "roofline_attainable_tflops": (
                            p.metrics.roofline_attainable_tflops
                        ),
                        "roofline_utilization_pct": (
                            p.metrics.roofline_utilization_pct
                        ),
                    },
                    "generation": p.generation,
                    "program_id": p.program_id,
                }
                for p in self.database.get_top_k(5)
            ],
        }

    # ------------------------------------------------------------------
    # Thin delegates to platform components
    # ------------------------------------------------------------------

    def _benchmark_pytorch_baseline(self, problem_file: Path) -> float:
        """Benchmark the eager reference implementation."""
        return self.benchmarker.benchmark_reference(problem_file)

    def _verify_initial_kernel(
        self,
        initial_kernel: str,
        problem_file: Path,
        test_code: list[str],
    ) -> bool:
        """Verify the initial kernel passes correctness before optimization."""
        return self.verifier.verify(initial_kernel, problem_file, test_code)

    def _benchmark_initial_kernel(
        self, initial_kernel: str, problem_file: Path
    ) -> float:
        """Benchmark the initial kernel before optimization begins."""
        return self.benchmarker.benchmark_kernel(initial_kernel, problem_file)

    def _load_resume_state(self, resume_from: Path | str) -> dict[str, Any] | None:
        """Load a prior run's state so optimization can continue from it.

        *resume_from* is a prior run's strategy dir (the one holding
        ``program_db.json`` and ``trajectory.jsonl``), or its parent. Returns
        the best program to seed the strategy with, all prior programs to
        re-seed the database, the next round number, or None if nothing usable.
        """
        base = Path(resume_from)
        db_path = base / "program_db.json"
        if not db_path.exists():
            # allow passing the run root; find the single strategy subdir's db
            cands = list(base.glob("*/program_db.json"))
            if not cands:
                self.logger.warning(f"resume: no program_db.json under {base}")
                return None
            db_path = cands[0]

        db = JSONProgramDatabase(db_path)
        db.load()
        programs = list(db.programs.values())
        finite = [p for p in programs if p.metrics.time_ms != float("inf")]
        if not finite:
            self.logger.warning(f"resume: no benchmarked programs in {db_path}")
            return None
        best = min(finite, key=lambda p: p.metrics.time_ms)

        # Continue round numbering past the prior run's last round. program_id
        # is like "r2_w0"; the generation field also carries the round.
        last_round = max((p.generation for p in programs), default=0)
        for p in programs:
            pid = p.program_id
            if pid.startswith("r") and "_" in pid:
                try:
                    last_round = max(last_round, int(pid[1:].split("_")[0]))
                except ValueError:
                    pass

        return {
            "kernel_code": best.kernel_code,
            "best_program": best,
            "best_ms": best.metrics.time_ms,
            "programs": programs,
            "next_round": last_round + 1,
        }

    @staticmethod
    def _record_round_trajectory(trajectory, round_num, best, baseline_ms):
        """Append one trajectory record for a finished round.

        ``best`` is the winning worker result for the round (or None when no
        worker succeeded). When present, its ``attempt`` dict carries the
        SOL / bottleneck / config detail the optimization worker measured.
        """
        if best is None:
            trajectory.record_round(
                round_num=round_num,
                time_ms=float("inf"),
                baseline_ms=baseline_ms,
                improvement_pct=0.0,
                compute_sol_pct=0.0,
                memory_sol_pct=0.0,
                combined_sol_pct=0.0,
                bottleneck="no_successful_worker",
                config_changes={},
                is_improvement=False,
                is_best=False,
                verified=False,
                kernel_file=None,
            )
            return

        attempt = best.get("attempt") or {}
        time_ms = best.get("time_ms", float("inf"))
        performance_fields = {
            "achieved_tflops": attempt.get("achieved_tflops"),
            "mfu_pct": attempt.get("mfu_pct"),
            "roofline_attainable_tflops": attempt.get(
                "roofline_attainable_tflops"
            ),
            "roofline_utilization_pct": attempt.get(
                "roofline_utilization_pct"
            ),
        }
        performance = (
            performance_fields
            if any(value is not None for value in performance_fields.values())
            else None
        )
        improvement = (
            ((baseline_ms - time_ms) / baseline_ms * 100.0)
            if baseline_ms and time_ms not in (None, float("inf")) and baseline_ms > 0
            else 0.0
        )
        trajectory.record_round(
            round_num=round_num,
            time_ms=time_ms,
            baseline_ms=baseline_ms,
            improvement_pct=improvement,
            compute_sol_pct=attempt.get("compute_sol_pct", 0.0),
            memory_sol_pct=attempt.get("memory_sol_pct", 0.0),
            combined_sol_pct=attempt.get("combined_sol_pct", 0.0),
            bottleneck=attempt.get("bottleneck_category", "unknown"),
            config_changes=attempt.get("config_changes", {}),
            is_improvement=bool(time_ms not in (None, float("inf")) and time_ms < baseline_ms),
            is_best=True,
            verified=True,
            kernel_file=None,
            performance=performance,
        )

    def _benchmark_pytorch_compile(self, problem_file: Path) -> float:
        """Benchmark the compiler-optimized reference."""
        return self.benchmarker.benchmark_reference_compiled(problem_file)

    def _run_workers(
        self,
        candidates: list[dict[str, Any]],
        round_num: int,
        problem_file: Path,
        test_code: list[str],
        pytorch_baseline: float,
    ) -> list[dict[str, Any]]:
        """Spawn workers for each candidate and collect results."""
        progress = get_progress()
        progress.emit(
            "agent2.workers",
            source="agent2.manager",
            message=f"round {round_num} workers running",
            strategy=self.strategy_name,
            round=round_num,
            worker_count=len(candidates),
        )
        kwargs = {
            "candidates": candidates,
            "round_num": round_num,
            "problem_file": problem_file,
            "test_code": test_code,
            "pytorch_baseline": pytorch_baseline,
            "shared_history": (
                self.shared_history[-self.history_size :] if self.shared_history else []
            ),
            "shared_reflexions": (
                self.shared_reflexions[-self.history_size :]
                if self.shared_reflexions
                else []
            ),
        }
        results = self.worker_runner.run_workers(**kwargs)
        progress.emit(
            "agent2.workers",
            source="agent2.manager",
            status="completed",
            message=f"round {round_num} workers returned",
            strategy=self.strategy_name,
            round=round_num,
            result_count=len(results),
            success_count=sum(1 for result in results if result.get("success")),
        )

        # Collect history and reflexions from worker results
        for r in results:
            if r.get("attempt"):
                self.shared_history.append(r["attempt"])
            if r.get("reflexion"):
                self.shared_reflexions.append(r["reflexion"])

        # Log errors from failed workers
        for r in results:
            if not r.get("success") and r.get("error"):
                self.logger.error(
                    f"Worker {r.get('worker_id')} failed: {r.get('error')}"
                )
                if r.get("traceback"):
                    self.logger.debug(f"Traceback:\n{r.get('traceback')}")

        return results
