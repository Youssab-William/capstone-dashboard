import os
from datetime import datetime
from typing import List
from ..interfaces import LLMProvider
from ..domain import TaskSpec, CompletionRecord
from ..config import PROVIDER_MODELS
import logging


class ClaudeProvider(LLMProvider):
    """
    Anthropic Claude provider.
    """

    def __init__(self, versions: List[str] = None) -> None:
        if versions is None:
            cfg = PROVIDER_MODELS.get("claude")
            versions = [v.id for v in cfg.versions] if cfg else ["claude-sonnet-4-20250514"]
        self._versions = versions
        self.logger = logging.getLogger(__name__)

    def name(self) -> str:
        return "claude"

    def versions(self) -> List[str]:
        return self._versions

    def generate(self, task: TaskSpec, version: str) -> CompletionRecord:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        msg = client.messages.create(model=version, max_tokens=1024, messages=[{"role": "user", "content": task.prompt}])
        text = "".join([c.text for c in msg.content if getattr(c, "type", "") == "text"]) or (msg.content[0].text if msg.content else "")
        usage = {"input_tokens": getattr(msg.usage, "input_tokens", 0), "output_tokens": getattr(msg.usage, "output_tokens", 0)}
        self.logger.debug(f"claude_call task_id={task.id} version={version} tone={task.tone} len={len(text)}")
        return CompletionRecord(
            run_id="",
            task_id=task.id,
            category=task.category,
            model=self.name(),
            version=version,
            tone=task.tone,
            prompt=task.prompt,
            response_text=text,
            usage=usage,
            created_at=datetime.utcnow(),
        )
