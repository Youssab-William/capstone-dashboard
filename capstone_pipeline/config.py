from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ProviderConfig:
    name: str
    versions: List[str]


@dataclass
class StorageConfig:
    root_dir: str = "data"


@dataclass
class PipelineConfig:
    experiment_name: str
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    storage: StorageConfig = field(default_factory=StorageConfig)
    schedule_cron: str = "0 3 * * 1"
