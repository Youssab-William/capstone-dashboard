from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
from ..interfaces import LLMProvider, MetricsEngine, AnalysisEngine, StorageRepository
from ..domain import TaskSpec, CompletionRecord
from ..runner.response_collector import ResponseCollector
from ..runner.progress import RunProgressTracker


class Pipeline:
    """
    Orchestrates the full workflow:
    - collect completions
    - compute metrics
    - run analysis
    """

    def __init__(self, storage: StorageRepository, metrics: MetricsEngine, analysis: AnalysisEngine) -> None:
        self.storage = storage
        self.metrics = metrics
        self.analysis = analysis
        self.logger = logging.getLogger(__name__)

    def _generate_run_id(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    def run_all(
        self,
        tasks: List[TaskSpec],
        providers: List[LLMProvider],
        run_id: Optional[str] = None,
        progress: Optional[RunProgressTracker] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full pipeline.

        - If run_id is None, a new id is generated.
        - If run_id is provided, the collector will *only* generate completions
          that do not already exist for that run (idempotent / retry-missing).
        - If a RunProgressTracker is provided, progress is persisted across
          the collect → metrics → analyze phases.
        """
        run_id = run_id or self._generate_run_id()
        self.logger.info("pipeline_run_start run_id=%s tasks=%s providers=%s", run_id, len(tasks), len(providers))

        try:
            # Collection phase
            if progress is not None:
                progress.set_status("running")
                progress.set_phase("collect")
            collector = ResponseCollector(self.storage)
            completions: List[CompletionRecord] = collector.run(tasks, providers, run_id, progress=progress)
            self.logger.info("completions_collected=%s run_id=%s", len(completions), run_id)

            # Metrics phase
            if progress is not None:
                progress.set_phase("metrics")
            metrics_rows = self.metrics.compute(completions)
            self.logger.info("metrics_rows=%s run_id=%s", len(metrics_rows), run_id)
            self.storage.save_rows(metrics_rows, "metrics", run_id)

            # Analysis phase
            if progress is not None:
                progress.set_phase("analyze")
            analysis_artifacts = self.analysis.analyze(metrics_rows)
            self.storage.save_rows([analysis_artifacts], "analysis", run_id)

            if progress is not None:
                progress.set_phase("done")
                progress.set_status("completed")
            self.logger.info("pipeline_run_completed run_id=%s", run_id)
            return {"run_id": run_id}
        except Exception as e:
            self.logger.exception("pipeline_run_failed run_id=%s error=%s", run_id, e)
            if progress is not None:
                progress.set_status("error", error=str(e))
            # Re-raise so callers can surface this appropriately.
            raise
