import os
from datetime import datetime
from typing import List
from ..interfaces import LLMProvider
from ..domain import TaskSpec, CompletionRecord
import logging


class GeminiProvider(LLMProvider):
    def __init__(self, versions: List[str] = None) -> None:
        self._versions = versions or ["gemini-2.5-pro"]
        self.logger = logging.getLogger(__name__)

    def name(self) -> str:
        return "gemini"

    def versions(self) -> List[str]:
        return self._versions

    def generate(self, task: TaskSpec, version: str) -> CompletionRecord:
        import google.generativeai as genai
        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GOOGLE_GEMINI_API_KEY")
            or ""
        )
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(version)
        self.logger.info(f"gemini_request task_id={task.id} version={version} tone={task.tone}")
        r = model.generate_content(task.prompt)
        text = getattr(r, "text", "")
        usage = {}
        um = getattr(r, "usage_metadata", None)
        if um is not None:
            usage["candidates_token_count"] = getattr(um, "candidates_token_count", 0)
        self.logger.info(f"gemini_ok task_id={task.id} version={version} tone={task.tone} tokens={usage.get('candidates_token_count', 0)} len={len(text)}")
        return CompletionRecord(
            run_id="",
            task_id=task.id,
            category=task.category,
            model=self.name(),
            version=version,
            tone=task.tone,
            prompt=task.prompt,
            response_text=text,
            usage=usage or None,
            created_at=datetime.utcnow(),
        )
