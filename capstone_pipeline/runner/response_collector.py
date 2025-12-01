from typing import List, Dict
from datetime import datetime
import logging
from ..interfaces import LLMProvider, StorageRepository
from ..domain import TaskSpec, CompletionRecord
from .progress import RunProgressTracker


class ResponseCollector:
    def __init__(self, storage: StorageRepository) -> None:
        self.storage = storage
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        tasks: List[TaskSpec],
        providers: List[LLMProvider],
        run_id: str,
        progress: RunProgressTracker | None = None,
    ) -> List[CompletionRecord]:
        """
        Collect completions for all tasks × providers × versions for a given run_id.

        If a RunProgressTracker is provided, this method will:
        - initialise total counts (including already existing completions)
        - increment per-model counters as new completions are produced.
        """
        rows: List[CompletionRecord] = []
        existing = self.storage.load_rows("completions", run_id)
        self.logger.info("existing_completions=%s run_id=%s", len(existing), run_id)
        seen = set((r.get("task_id"), r.get("model"), r.get("version"), r.get("tone")) for r in existing)

        # Progress initialisation: compute theoretical totals and already completed per model.
        if progress is not None:
            per_model_total: Dict[str, int] = {}
            for provider in providers:
                m = provider.name()
                per_model_total[m] = per_model_total.get(m, 0) + len(tasks) * len(provider.versions())
            already_completed: Dict[str, int] = {}
            for r in existing:
                m = r.get("model") or ""
                if not m:
                    continue
                already_completed[m] = already_completed.get(m, 0) + 1
            total_expected = sum(per_model_total.values())
            progress.init_collect(total_expected, per_model_total, already_completed)

        for task in tasks:
            for provider in providers:
                model_name = provider.name()
                for v in provider.versions():
                    key = (task.id, model_name, v, task.tone)
                    if key in seen:
                        self.logger.debug("skip_existing key=%s", key)
                        continue
                    try:
                        cr = provider.generate(task, v)
                        cr.run_id = run_id
                        self.logger.debug(
                            "generated task_id=%s model=%s version=%s tone=%s len=%s",
                            task.id,
                            model_name,
                            v,
                            task.tone,
                            len(cr.response_text or ""),
                        )
                        rows.append(cr)
                        if progress is not None:
                            progress.increment_collect(model_name, success=True)
                    except Exception as e:
                        # Treat all non-successful calls (e.g., non-2xx HTTP responses,
                        # quota errors, invalid parameters) as "skipped" for the
                        # purposes of analysis: we log them and mark them as failed
                        # in progress, but we do NOT create a synthetic CompletionRecord.
                        self.logger.warning(
                            "provider_error task_id=%s model=%s version=%s tone=%s error=%s",
                            task.id,
                            model_name,
                            v,
                            task.tone,
                            e,
                        )
                        if progress is not None:
                            progress.increment_collect(model_name, success=False)
        self._persist(rows)
        return rows

    def _persist(self, rows: List[CompletionRecord]) -> None:
        payload = []
        for r in rows:
            payload.append({
                "run_id": r.run_id,
                "task_id": r.task_id,
                "category": getattr(r, "category", ""),
                "model": r.model,
                "version": r.version,
                "tone": r.tone,
                "prompt": r.prompt,
                "response_text": r.response_text,
                "usage": r.usage,
                "created_at": r.created_at.isoformat(),
            })
        self.logger.info(f"persisting_completions rows={len(payload)}")
        self.storage.save_rows(payload, "completions", payload[0]["run_id"] if payload else None)
