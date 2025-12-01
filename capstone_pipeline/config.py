from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ProviderConfig:
    """
    Backwards-compatible provider configuration used for higher-level
    experiment definitions (not currently wired into the main pipeline).
    """
    name: str
    versions: List[str]


@dataclass
class StorageConfig:
    """
    Simple storage configuration – currently only the root directory.
    """
    root_dir: str = "data"


@dataclass
class PipelineConfig:
    """
    High-level pipeline configuration placeholder.
    """
    experiment_name: str
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    storage: StorageConfig = field(default_factory=StorageConfig)
    schedule_cron: str = "0 3 * * 1"


# ---------------------------------------------------------------------------
# Model / provider configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelVersionConfig:
    """
    Configuration for a specific model version exposed by a provider.

    - id: exact API model identifier (e.g. "gpt-4.1")
    - label: human-friendly name for UI display (e.g. "GPT‑4.1")
    """
    id: str
    label: str


@dataclass
class ProviderModelConfig:
    """
    Configuration for all model versions of a logical provider.

    - provider_key: short key used in data/analysis (e.g. "gpt", "claude")
    - versions: list of ModelVersionConfig entries
    """
    provider_key: str
    versions: List[ModelVersionConfig] = field(default_factory=list)


# Central registry of the model versions we want to expose. This keeps
# the provider implementations lean and makes it trivial to add/remove
# versions without touching the analysis code.
PROVIDER_MODELS: Dict[str, ProviderModelConfig] = {
    "gpt": ProviderModelConfig(
        provider_key="gpt",
        versions=[
            # Stable GPT-4
            ModelVersionConfig("gpt-4o", "GPT‑4o"),
            # Newer GPT-5 family
            ModelVersionConfig("gpt-5", "GPT‑5"),
            ModelVersionConfig("gpt-5.1", "GPT‑5.1"),
        ],
    ),
    "claude": ProviderModelConfig(
        provider_key="claude",
        versions=[
            # Claude 4.5 family (ids provided from your account)
            ModelVersionConfig("claude-opus-4-5-20251101", "Claude Opus 4.5"),
            ModelVersionConfig("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5"),
            # Existing Sonnet 4 snapshot (kept for backwards-compatibility)
            ModelVersionConfig("claude-sonnet-4-20250514", "Claude Sonnet 4 (2025‑05‑14)"),
        ],
    ),
    "gemini": ProviderModelConfig(
        provider_key="gemini",
        versions=[
            ModelVersionConfig("gemini-2.5-pro", "Gemini 2.5 Pro"),
            # New Gemini 3 Pro Preview (full model name as required by the API)
            ModelVersionConfig("models/gemini-3-pro-preview", "Gemini 3 Pro Preview"),
        ],
    ),
}

