# Capstone Pipeline - LLM Tone Effects Analysis

## Overview

This project is a comprehensive pipeline for analyzing how different tones (Polite vs. Threatening) affect LLM responses across multiple models (GPT, Claude, DeepSeek, Gemini). The system collects responses, computes metrics, and performs statistical analysis.

## Architecture

### Core Components

1. **Providers** (`capstone_pipeline/providers/`)
   - `GPTProvider`: OpenAI GPT models
   - `ClaudeProvider`: Anthropic Claude models
   - `DeepSeekProvider`: DeepSeek API
   - `GeminiProvider`: Google Gemini models
   - Each provider implements the `LLMProvider` interface

2. **Storage** (`capstone_pipeline/storage/`)
   - `JsonlStorage`: Stores data in JSONL format
   - Organizes data by table (completions, metrics, analysis) and run_id partitions

3. **Metrics Engine** (`capstone_pipeline/metrics/`)
   - `ScriptMetricsEngine`: Computes metrics using legacy analysis modules
   - Metrics include:
     - Sentiment scores (VADER)
     - Toxicity scores (RoBERTa-based)
     - Politeness scores
     - Refusal/disclaimer detection
     - Response length

4. **Analysis Engine** (`capstone_pipeline/analysis/`)
   - `ScriptAnalysisEngine`: Performs statistical analysis
   - Generates summaries, deltas, correlations, paired tests
   - Analyzes by model, tone, category

5. **Workflow** (`capstone_pipeline/workflow/`)
   - `Pipeline`: Orchestrates the full pipeline
   - `ResponseCollector`: Collects responses from all providers

### Data Flow

```
Tasks (prompts.txt) 
  → ResponseCollector 
    → LLM Providers (GPT, Claude, DeepSeek, Gemini)
      → Completions Storage (data/completions/{run_id}.jsonl)
        → Metrics Engine
          → Metrics Storage (data/metrics/{run_id}.jsonl)
            → Analysis Engine
              → Analysis Storage (data/analysis/{run_id}.jsonl)
```

### Command-Line Interface

The main entry point is `capstone_pipeline/app.py` with the following commands:

- `run`: Execute the full pipeline (collect → metrics → analyze)
- `collect`: Only collect responses from LLMs
- `metrics`: Compute metrics for existing completions (requires --run_id)
- `analyze`: Analyze existing metrics (requires --run_id)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

API keys are stored in `data/keys.json`. The file should contain:
```json
{
  "openai_api_key": "your-key",
  "anthropic_api_key": "your-key",
  "deepseek_api_key": "your-key",
  "gemini_api_key": "your-key",
  "google_api_key": "your-key"
}
```

Alternatively, you can set environment variables:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `DEEPSEEK_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`

### 3. Run the Pipeline

#### Full Pipeline (Recommended)
```bash
python -m capstone_pipeline.app run --keys_file data/keys.json --prompts_file prompts.txt
```

#### Step-by-Step
```bash
# Step 1: Collect responses
python -m capstone_pipeline.app collect --keys_file data/keys.json --prompts_file prompts.txt

# Step 2: Compute metrics (replace RUN_ID with actual run_id from step 1)
python -m capstone_pipeline.app metrics --run_id RUN_ID

# Step 3: Analyze metrics
python -m capstone_pipeline.app analyze --run_id RUN_ID
```

### 4. Run the Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

## Project Structure

```
capstone_pipeline/
├── app.py                 # CLI entry point
├── config.py              # Configuration classes
├── domain.py              # Domain models (TaskSpec, CompletionRecord)
├── interfaces.py          # Protocol definitions
├── providers/             # LLM provider implementations
├── storage/               # Data storage (JSONL)
├── metrics/               # Metrics computation
├── analysis/              # Statistical analysis
├── workflow/              # Pipeline orchestration
└── runner/                # Response collection

data/
├── completions/           # Raw LLM responses
├── metrics/               # Computed metrics
├── analysis/              # Analysis results
└── keys.json             # API keys

prompts.txt                # Task definitions (JSON format)
```

## Task Format

Tasks are defined in `prompts.txt` as JSON array with:
- `TaskID`: Unique identifier
- `TaskDescription`: Description of the task
- `PromptTone`: "Polite" or "Threatening"
- `PromptText`: The actual prompt text

## Output

Results are stored in `data/` directory:
- `completions/{run_id}.jsonl`: Raw responses from LLMs
- `metrics/{run_id}.jsonl`: Computed metrics for each response
- `analysis/{run_id}.jsonl`: Statistical analysis results

Each run gets a unique `run_id` in format `YYYYMMDD-HHMMSS`.

## Notes

- The pipeline skips existing completions to avoid duplicate API calls
- Metrics are computed using legacy modules from `legacy/phase_2_legacy_code/modules/`
- The dashboard (`dashboard/streamlit_app.py`) provides a UI to view results and run new analyses

