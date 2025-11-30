from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime


@dataclass
class TaskSpec:
    id: str
    category: str
    tone: str
    prompt: str


@dataclass
class CompletionRecord:
    run_id: str
    task_id: str
    category: str
    model: str
    version: str
    tone: str
    prompt: str
    response_text: str
    usage: Optional[Dict[str, float]]
    created_at: datetime
