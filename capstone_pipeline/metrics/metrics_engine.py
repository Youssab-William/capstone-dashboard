from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
import importlib.util
import sys
from pathlib import Path
from ..interfaces import MetricsEngine
from ..domain import CompletionRecord
import logging


class ScriptMetricsEngine(MetricsEngine):
    def __init__(self) -> None:
        root = Path.cwd() / "legacy" / "phase_2_legacy_code" / "modules"
        self.sentiment = self._load_class(root / "sentiment_analyzer.py", "SentimentAnalyzer")
        self.toxicity = self._load_class(root / "toxicity_analyzer.py", "ToxicityAnalyzer")
        self.politeness = self._load_class(root / "politeness_analyzer.py", "PolitenessAnalyzer")
        self.refusal = self._load_class(root / "refusal_disclaimer_detector.py", "RefusalDisclaimerDetector")
        self.logger = logging.getLogger(__name__)
        self.logger.info("metrics_engine_initialized")

    def _load_class(self, path: Path, class_name: str):
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, class_name)()

    def compute(self, completions: List[CompletionRecord]) -> List[Dict[str, Any]]:
        if not completions:
            return []
        self.logger.info(f"metrics_compute input_rows={len(completions)}")
        rows = []
        def _int_from_usage(u, key):
            if not isinstance(u, dict):
                return 0
            v = u.get(key)
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, (int, float)):
                        return int(vv)
                    if isinstance(vv, str):
                        try:
                            return int(float(vv))
                        except Exception:
                            continue
                return 0
            if isinstance(v, str):
                try:
                    return int(float(v))
                except Exception:
                    return 0
            return 0
        for c in completions:
            resp_len = _int_from_usage(c.usage or {}, "completion_tokens") or _int_from_usage(c.usage or {}, "output_tokens")
            rows.append({
                "TaskID": c.task_id,
                "TaskCategory": c.category,
                "PromptText": c.prompt,
                "ResponseText": c.response_text,
                "Model": c.model,
                "Version": c.version,
                "PromptTone": c.tone,
                "ResponseLength": resp_len or (c.usage or {}).get("candidates_token_count") or 0,
            })
        df = pd.DataFrame(rows)
        self.logger.info(f"metrics_dataframe rows={len(df)} cols={list(df.columns)}")
        df = self.sentiment.analyze_dataframe(df, "PromptText", "ResponseText")
        df = self.politeness.analyze_dataframe(df, "PromptText", "ResponseText")
        df = self.refusal.analyze_dataframe(df, "PromptText", "ResponseText")
        df = self.toxicity.analyze_dataframe(df, "PromptText", "ResponseText")
        out = df.to_dict(orient="records")
        self.logger.info(f"metrics_output rows={len(out)}")
        for r in out:
            r["created_at"] = datetime.utcnow().isoformat()
        return out
