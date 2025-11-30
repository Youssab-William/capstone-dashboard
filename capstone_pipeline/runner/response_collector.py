from typing import List
from datetime import datetime
import logging
from ..interfaces import LLMProvider, StorageRepository
from ..domain import TaskSpec, CompletionRecord


class ResponseCollector:
    def __init__(self, storage: StorageRepository) -> None:
        self.storage = storage
        self.logger = logging.getLogger(__name__)

    def run(self, tasks: List[TaskSpec], providers: List[LLMProvider], run_id: str) -> List[CompletionRecord]:
        rows: List[CompletionRecord] = []
        existing = self.storage.load_rows("completions", run_id)
        self.logger.info(f"existing_completions={len(existing)} run_id={run_id}")
        seen = set((r.get("task_id"), r.get("model"), r.get("version"), r.get("tone")) for r in existing)
        for task in tasks:
            for provider in providers:
                for v in provider.versions():
                    key = (task.id, provider.name(), v, task.tone)
                    if key in seen:
                        self.logger.debug(f"skip_existing key={key}")
                        continue
                    try:
                        cr = provider.generate(task, v)
                        cr.run_id = run_id
                        self.logger.debug(f"generated task_id={task.id} model={provider.name()} version={v} tone={task.tone}")
                    except Exception as e:
                        self.logger.warning(f"provider_error task_id={task.id} model={provider.name()} version={v} tone={task.tone} error={e}")
                        cr = CompletionRecord(
                            run_id=run_id,
                            task_id=task.id,
                            category=task.category,
                            model=provider.name(),
                            version=v,
                            tone=task.tone,
                            prompt=task.prompt,
                            response_text=str(e),
                            usage=None,
                            created_at=datetime.utcnow(),
                        )
                    rows.append(cr)
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
