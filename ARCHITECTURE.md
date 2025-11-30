# Architecture Overview

## System Design

The capstone pipeline is built with a modular, protocol-based architecture that separates concerns into distinct components.

## Core Architecture Principles

1. **Protocol-Based Design**: Uses Python `Protocol` types to define interfaces
2. **Dependency Injection**: Components receive dependencies through constructors
3. **Separation of Concerns**: Each module has a single responsibility
4. **Data-Driven**: Configuration and tasks are defined in JSON files

## Component Details

### 1. Domain Models (`domain.py`)

**TaskSpec**: Represents a task to be sent to LLMs
- `id`: Unique task identifier (e.g., "ProgrammingHelpTask1")
- `category`: Extracted category from task ID (e.g., "ProgrammingHelp")
- `tone`: "Polite" or "Threatening"
- `prompt`: The actual prompt text

**CompletionRecord**: Represents a response from an LLM
- `run_id`: Unique identifier for the pipeline run
- `task_id`: Reference to the original task
- `category`: Task category
- `model`: Provider name (gpt, claude, deepseek, gemini)
- `version`: Model version (e.g., "gpt-4o", "claude-sonnet-4-20250514")
- `tone`: Prompt tone used
- `prompt`: Original prompt
- `response_text`: LLM response
- `usage`: Token usage information
- `created_at`: Timestamp

### 2. Interfaces (`interfaces.py`)

Defines protocols that components must implement:

**LLMProvider**: Interface for LLM providers
- `name() -> str`: Returns provider name
- `versions() -> List[str]`: Returns available model versions
- `generate(task: TaskSpec, version: str) -> CompletionRecord`: Generates a response

**MetricsEngine**: Interface for metrics computation
- `compute(completions: List[CompletionRecord]) -> List[Dict]`: Computes metrics

**AnalysisEngine**: Interface for statistical analysis
- `analyze(metrics_rows: List[Dict]) -> Dict`: Performs analysis

**StorageRepository**: Interface for data persistence
- `save_rows(rows, table, partition)`: Saves data
- `load_rows(table, partition)`: Loads data
- `exists(table, partition)`: Checks existence

### 3. Providers (`providers/`)

Each provider implements the `LLMProvider` protocol:

**GPTProvider** (`gpt.py`)
- Uses OpenAI's Python SDK
- Default version: "gpt-4o"
- Environment variable: `OPENAI_API_KEY`

**ClaudeProvider** (`claude.py`)
- Uses Anthropic's Python SDK
- Default version: "claude-sonnet-4-20250514"
- Environment variable: `ANTHROPIC_API_KEY`

**DeepSeekProvider** (`deepseek.py`)
- Uses REST API via `requests`
- Default version: "deepseek-chat"
- Environment variable: `DEEPSEEK_API_KEY`

**GeminiProvider** (`gemini.py`)
- Uses Google's Generative AI SDK
- Default version: "gemini-2.5-pro"
- Environment variable: `GEMINI_API_KEY` or `GOOGLE_API_KEY`

### 4. Storage (`storage/repository.py`)

**JsonlStorage**: Implements `StorageRepository`
- Stores data as JSONL (JSON Lines) files
- Organizes by table (completions, metrics, analysis) and partition (run_id)
- File structure: `{data_dir}/{table}/{partition}.jsonl`
- Appends to files (supports incremental writes)

### 5. Metrics Engine (`metrics/metrics_engine.py`)

**ScriptMetricsEngine**: Implements `MetricsEngine`
- Loads legacy analysis modules dynamically
- Modules used:
  - `SentimentAnalyzer`: VADER sentiment analysis
  - `ToxicityAnalyzer`: RoBERTa-based toxicity detection
  - `PolitenessAnalyzer`: Politeness scoring
  - `RefusalDisclaimerDetector`: Detects refusals and disclaimers

**Metrics Computed:**
- `Response_SentimentScore`: Compound sentiment score (-1 to 1)
- `Response_ValidatedPolitenessScore`: Politeness score
- `RoBERTa_Response_ToxicityScore`: Toxicity score
- `ResponseLength`: Response length in tokens
- `Response_RefusalFlag`: Boolean refusal detection
- `Response_DisclaimerFlag`: Boolean disclaimer detection
- `Response_ValidatedStrategies`: Politeness strategies used

### 6. Analysis Engine (`analysis/analysis_engine.py`)

**ScriptAnalysisEngine**: Implements `AnalysisEngine`
- Performs statistical analysis on metrics
- Uses pandas for data manipulation

**Analysis Output:**
- `summary`: Mean metrics by model/version/tone
- `deltas`: Differences between Polite and Threatening tones
- `categories`: Category-level aggregations
- `safety`: Refusal and disclaimer rates
- `strategies`: Politeness strategy frequencies
- `correlations`: Correlations between metrics
- `paired_tests`: Paired comparisons (Polite vs Threatening)
- `highlights`: Significant findings

### 7. Workflow (`workflow/pipeline.py`)

**Pipeline**: Main orchestrator
- Coordinates the full pipeline execution
- Generates unique run_id (timestamp-based)
- Calls ResponseCollector, MetricsEngine, and AnalysisEngine in sequence

**ResponseCollector** (`runner/response_collector.py`):
- Iterates through all tasks and providers
- Skips existing completions (deduplication)
- Handles errors gracefully (stores error messages)
- Persists completions to storage

### 8. CLI (`app.py`)

**Commands:**
- `run`: Full pipeline (collect → metrics → analyze)
- `collect`: Only collect responses
- `metrics`: Compute metrics for existing completions
- `analyze`: Analyze existing metrics

**Configuration:**
- `--data_dir`: Data directory (default: "data")
- `--run_id`: Specific run ID (required for metrics/analyze)
- `--prompts_file`: Tasks file (default: "prompts.txt")
- `--keys_file`: API keys file
- `--log_level`: Logging level
- `--log_file`: Log file path

## Data Flow

```
1. Parse prompts.txt → List[TaskSpec]
2. For each TaskSpec:
   - For each Provider:
     - For each Version:
       - Generate response → CompletionRecord
       - Save to data/completions/{run_id}.jsonl
3. Load all CompletionRecords
4. Compute metrics → List[Dict]
   - Save to data/metrics/{run_id}.jsonl
5. Load metrics
6. Analyze metrics → Dict
   - Save to data/analysis/{run_id}.jsonl
```

## Error Handling

- **Provider Errors**: Caught and stored as error messages in CompletionRecord
- **Missing Data**: Gracefully handles missing files/partitions
- **API Failures**: Logged and stored for debugging

## Extensibility

### Adding a New Provider

1. Create a new file in `providers/`
2. Implement `LLMProvider` protocol
3. Add to provider list in `app.py`

### Adding a New Metric

1. Create analyzer in `legacy/phase_2_legacy_code/modules/`
2. Load in `ScriptMetricsEngine.__init__()`
3. Apply in `ScriptMetricsEngine.compute()`

### Customizing Analysis

Modify `ScriptAnalysisEngine.analyze()` to add new statistical tests or visualizations.

## Performance Considerations

- **Deduplication**: Skips existing completions to avoid redundant API calls
- **Incremental Processing**: Can run metrics/analysis separately
- **Batch Processing**: Processes all tasks in a single run
- **Storage**: JSONL format allows streaming and appending

## Testing Strategy

- Unit tests: Test each component in isolation
- Integration tests: Test full pipeline with mock providers
- End-to-end tests: Test with real providers (requires API keys)

