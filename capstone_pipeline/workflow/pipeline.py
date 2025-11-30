from typing import List, Dict, Any
from datetime import datetime
import logging
from ..interfaces import LLMProvider, MetricsEngine, AnalysisEngine, StorageRepository
from ..domain import TaskSpec
from ..runner.response_collector import ResponseCollector


class Pipeline:
    def __init__(self, storage: StorageRepository, metrics: MetricsEngine, analysis: AnalysisEngine) -> None:
        self.storage = storage
        self.metrics = metrics
        self.analysis = analysis
        self.logger = logging.getLogger(__name__)

    def run_all(self, tasks: List[TaskSpec], providers: List[LLMProvider]) -> Dict[str, Any]:
        run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.logger.info(f"run_id={run_id} tasks={len(tasks)} providers={len(providers)}")
        completions = ResponseCollector(self.storage).run(tasks, providers, run_id)
        self.logger.info(f"completions_collected={len(completions)}")
        metrics_rows = self.metrics.compute(completions)
        self.logger.info(f"metrics_rows={len(metrics_rows)}")
        self.storage.save_rows(metrics_rows, "metrics", run_id)
        analysis_artifacts = self.analysis.analyze(metrics_rows)
        self.storage.save_rows([analysis_artifacts], "analysis", run_id)
        self.logger.info("pipeline_run_completed")
        return {"run_id": run_id}
