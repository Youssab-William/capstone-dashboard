import os
import json
import requests
from datetime import datetime
from typing import List
from ..interfaces import LLMProvider
from ..domain import TaskSpec, CompletionRecord
import logging


class DeepSeekProvider(LLMProvider):
    def __init__(self, versions: List[str] = None) -> None:
        self._versions = versions or ["deepseek-chat"]
        self.logger = logging.getLogger(__name__)

    def name(self) -> str:
        return "deepseek"

    def versions(self) -> List[str]:
        return self._versions

    def generate(self, task: TaskSpec, version: str) -> CompletionRecord:
        api_key = (
            os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("DEEPSEEK_KEY")
            or os.environ.get("DEEPSEEK_TOKEN")
            or os.environ.get("API_KEY_DEEPSEEK")
            or ""
        )
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": version,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": task.prompt},
            ],
            "stream": False,
        }
        url = "https://api.deepseek.com/chat/completions"
        self.logger.info(f"deepseek_request task_id={task.id} version={version} tone={task.tone}")
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        raw_usage = body.get("usage", {})
        def _first_numeric(val):
            if isinstance(val, (int, float)):
                return int(val)
            if isinstance(val, str):
                try:
                    return int(float(val))
                except Exception:
                    return None
            if isinstance(val, dict):
                for v in val.values():
                    if isinstance(v, (int, float)):
                        return int(v)
                    if isinstance(v, str):
                        try:
                            return int(float(v))
                        except Exception:
                            continue
            return None
        usage = {}
        for k in ("completion_tokens", "output_tokens", "total_tokens", "prompt_tokens"):
            if k in raw_usage:
                n = _first_numeric(raw_usage.get(k))
                if n is not None:
                    usage[k] = n
        if not usage:
            usage = None
        self.logger.info(f"deepseek_ok task_id={task.id} version={version} tone={task.tone} tokens={(usage or {}).get('completion_tokens', 0)}")
        self.logger.debug(f"deepseek_call task_id={task.id} version={version} tone={task.tone} len={len(content)}")
        return CompletionRecord(
            run_id="",
            task_id=task.id,
            category=task.category,
            model=self.name(),
            version=version,
            tone=task.tone,
            prompt=task.prompt,
            response_text=content,
            usage={k: float(v) for k, v in usage.items()} if usage else None,
            created_at=datetime.utcnow(),
        )
