import os
from datetime import datetime
from typing import List
from ..interfaces import LLMProvider
from ..domain import TaskSpec, CompletionRecord
import logging


class GPTProvider(LLMProvider):
    def __init__(self, versions: List[str] = None) -> None:
        self._versions = versions or ["gpt-4o"]
        self.logger = logging.getLogger(__name__)

    def name(self) -> str:
        return "gpt"

    def versions(self) -> List[str]:
        return self._versions

    def generate(self, task: TaskSpec, version: str) -> CompletionRecord:
        from openai import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model=version,
            messages=[{"role": "user", "content": task.prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        content = r.choices[0].message.content
        usage = r.usage.model_dump() if hasattr(r, "usage") else None
        self.logger.debug(f"gpt_call task_id={task.id} version={version} tone={task.tone} len={len(content)}")
        return CompletionRecord(
            run_id="",
            task_id=task.id,
            category=task.category,
            model=self.name(),
            version=version,
            tone=task.tone,
            prompt=task.prompt,
            response_text=content,
            usage=usage,
            created_at=datetime.utcnow(),
        )
