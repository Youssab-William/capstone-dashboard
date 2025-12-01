import os
from datetime import datetime, timezone
from typing import List
from ..interfaces import LLMProvider
from ..domain import TaskSpec, CompletionRecord
from ..config import PROVIDER_MODELS
import logging


class GPTProvider(LLMProvider):
    """
    OpenAI GPT provider.

    - name(): short provider key used in analysis ("gpt")
    - versions(): list of concrete API model ids (e.g. "gpt-4o", "gpt-4.1")
    """

    def __init__(self, versions: List[str] = None) -> None:
        if versions is None:
            cfg = PROVIDER_MODELS.get("gpt")
            versions = [v.id for v in cfg.versions] if cfg else ["gpt-4o"]
        self._versions = versions
        self.logger = logging.getLogger(__name__)

    def name(self) -> str:
        return "gpt"

    def versions(self) -> List[str]:
        return self._versions

    def generate(self, task: TaskSpec, version: str) -> CompletionRecord:
        """
        Call the OpenAI Chat Completions API.

        Note: Some of the newer models (e.g. GPT‑5 family) do not support the
        legacy `max_tokens` parameter and instead expect different limits
        (e.g. `max_completion_tokens` on the Responses API). To keep this
        provider broadly compatible across models and SDK versions we do not
        pass an explicit max token parameter here and rely on the model
        defaults. If you want to constrain length further, you can wrap the
        prompt or adjust this call to use the Responses API.
        """
        from openai import OpenAI

        client = OpenAI()
        # Some of the newest models (e.g. GPT‑5 family) only support the
        # default temperature. To keep this compatible across models, we
        # omit the temperature parameter and rely on the model default.
        r = client.chat.completions.create(
            model=version,
            messages=[{"role": "user", "content": task.prompt}],
        )
        content = r.choices[0].message.content
        usage = r.usage.model_dump() if hasattr(r, "usage") else None
        self.logger.debug(
            "gpt_call task_id=%s version=%s tone=%s len=%s",
            task.id,
            version,
            task.tone,
            len(content or ""),
        )
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
            created_at=datetime.now(timezone.utc),
        )
