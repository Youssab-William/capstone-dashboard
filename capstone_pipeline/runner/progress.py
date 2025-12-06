import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


logger = logging.getLogger(__name__)


@dataclass
class RunProgress:
    """
    Lightweight representation of the state of a single pipeline run.

    This is intentionally simple and stored as JSON on disk so that the
    Streamlit frontend can easily read it for live progress updates.
    """

    run_id: str
    created_at: str
    updated_at: str
    status: str  # "pending", "running", "completed", "error"
    phase: str   # "collect", "metrics", "analyze", "done"
    total_prompts: int = 0
    # per_model[model] = {"total": int, "completed": int, "failed": int}
    per_model: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Optional: selection metadata from the UI, used by the dashboard
    # to reconstruct per-model / per-version expectations for a run.
    selection: Dict[str, dict] = field(default_factory=dict)
    error: str = ""
    # GitHub commit status: "pending", "success", "failed", "skipped", "error"
    github_commit_status: str = ""
    github_commit_message: str = ""


class RunProgressTracker:
    """
    Helper for persisting run progress to a JSON file on disk.

    This class is deliberately conservative: every update overwrites the
    JSON file, trading a tiny bit of I/O for very robust and debuggable
    state.
    """

    def __init__(self, root_dir: str, run_id: str) -> None:
        self.root_dir = root_dir
        self.run_id = run_id
        self.path = Path(root_dir) / "logs" / f"run_{run_id}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("RunProgressTracker initialized root_dir=%s run_id=%s path=%s", root_dir, run_id, self.path)

    # --- internal helpers -------------------------------------------------

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load(self) -> RunProgress:
        if not self.path.exists():
            # New progress object with minimal defaults
            now = self._now_iso()
            return RunProgress(
                run_id=self.run_id,
                created_at=now,
                updated_at=now,
                status="pending",
                phase="collect",
            )
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Be defensive about missing fields so that older JSON files
            # (without newer attributes) do not break deserialization.
            default = asdict(
                RunProgress(
                    run_id=self.run_id,
                    created_at=self._now_iso(),
                    updated_at=self._now_iso(),
                    status="pending",
                    phase="collect",
                )
            )
            default.update(data)
            return RunProgress(**default)
        except Exception as e:
            logger.warning("RunProgressTracker load_error run_id=%s path=%s error=%s", self.run_id, self.path, e)
            # In case of corruption, start fresh but keep run_id
            now = self._now_iso()
            return RunProgress(
                run_id=self.run_id,
                created_at=now,
                updated_at=now,
                status="pending",
                phase="collect",
            )

    def _save(self, progress: RunProgress) -> None:
        progress.updated_at = self._now_iso()
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(progress), f, ensure_ascii=False, indent=2)
        tmp_path.replace(self.path)
        logger.debug(
            "RunProgressTracker saved run_id=%s status=%s phase=%s total_prompts=%s",
            progress.run_id,
            progress.status,
            progress.phase,
            progress.total_prompts,
        )

    # --- public API -------------------------------------------------------

    def set_selection(self, selection: Dict[str, dict]) -> None:
        """
        Persist the provider/model/versions selection used for this run.

        This allows the dashboard to reconstruct per-model/per-version
        expectations even after a Streamlit session is restarted.
        """
        progress = self._load()
        progress.selection = selection or {}
        self._save(progress)

    def init_collect(self, total_prompts: int, per_model_totals: Dict[str, int], already_completed: Dict[str, int]) -> None:
        """
        Initialize collection phase metadata for a run.

        total_prompts: theoretical number of completions (tasks × versions × models),
        per_model_totals: theoretical totals per model,
        already_completed: completions that already exist for this run (e.g. when retrying).
        """
        progress = self._load()
        progress.status = "running"
        progress.phase = "collect"
        progress.total_prompts = int(total_prompts)
        progress.per_model = {}
        for model, total in per_model_totals.items():
            completed = int(already_completed.get(model, 0))
            progress.per_model[model] = {
                "total": int(total),
                "completed": completed,
                "failed": 0,
            }
        self._save(progress)

    def increment_collect(self, model: str, success: bool) -> None:
        """
        Increment counters for a completed collection attempt.
        """
        progress = self._load()
        if model not in progress.per_model:
            # Should not happen, but make it robust.
            progress.per_model[model] = {"total": 0, "completed": 0, "failed": 0}
        key = "completed" if success else "failed"
        progress.per_model[model][key] = int(progress.per_model[model].get(key, 0)) + 1
        self._save(progress)

    def set_phase(self, phase: str) -> None:
        progress = self._load()
        progress.phase = phase
        self._save(progress)

    def set_status(self, status: str, error: str = "") -> None:
        progress = self._load()
        progress.status = status
        if error:
            progress.error = error
        self._save(progress)

    def set_github_commit_status(self, status: str, message: str = "") -> None:
        """
        Set the GitHub commit status and optional message.
        Status values: "pending", "success", "failed", "skipped", "error"
        """
        progress = self._load()
        progress.github_commit_status = status
        progress.github_commit_message = message
        self._save(progress)


