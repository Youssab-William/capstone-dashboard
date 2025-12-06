# LLM Tone Effects Dashboard

A comprehensive dashboard for analyzing how different LLM models respond to varying prompt tones (Polite vs. Threatening). This tool systematically tests multiple LLM models and versions, collects their responses, computes linguistic metrics, and performs statistical analysis to understand tone-based behavioral differences.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Dashboard User Guide](#dashboard-user-guide)
  - [Sidebar Navigation](#sidebar-navigation)
  - [Run Monitor](#run-monitor)
  - [Dashboard View](#dashboard-view)
- [Understanding the Metrics](#understanding-the-metrics)
- [Analysis Workflow](#analysis-workflow)
- [Technical Architecture](#technical-architecture)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What This Dashboard Does

This dashboard automates the process of:
1. **Collecting responses** from multiple LLM models (GPT, Claude, Gemini, DeepSeek)
2. **Testing tone effects** by sending identical tasks with different prompt tones
3. **Computing metrics** including sentiment, politeness, toxicity, and response characteristics
4. **Performing statistical analysis** to identify significant differences between tone conditions
5. **Visualizing results** through interactive charts and tables

### Key Features

✅ **Multi-Model Testing** - Test GPT, Claude, Gemini, and DeepSeek models simultaneously
✅ **Real-Time Monitoring** - Watch live progress as responses are collected
✅ **Comprehensive Metrics** - 40+ linguistic and behavioral metrics per response
✅ **Statistical Analysis** - Paired t-tests, correlations, and delta analysis
✅ **Interactive Visualizations** - Filter and explore results across 8 analysis tabs
✅ **Data Persistence** - Optional GitHub integration for cross-deployment data retention

---

## Getting Started

### Local Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure API Keys**:
Create `data/keys.json` with your LLM API keys:
```json
{
  "openai_api_key": "sk-...",
  "anthropic_api_key": "sk-ant-...",
  "gemini_api_key": "AI...",
  "deepseek_api_key": "sk-..."
}
```

3. **Run the Dashboard**:
```bash
streamlit run dashboard/streamlit_app.py
```

4. **Access**: Open your browser to `http://localhost:8501`

---

## Dashboard User Guide

The dashboard has **two main views** that you can switch between using the sidebar.

### Sidebar Navigation

At the top of the sidebar, you'll always see:

**View Selector (Radio Buttons)**
- 🏠 **Dashboard** - View and analyze completed runs
- 📊 **Run Monitor** - Configure and start new analysis runs

Simply click the radio button to switch between views.

---

## Run Monitor

The Run Monitor is your control center for configuring, starting, and monitoring analysis runs.

### Sidebar: Run Configuration

When in Run Monitor view, the sidebar displays expandable sections for each LLM provider:

#### 🤖 **DeepSeek**
Click to expand and configure:
- ☑️ **Enable DeepSeek** - Check to include DeepSeek models in your run
- Currently supports: `deepseek-chat`

#### 💬 **ChatGPT (OpenAI)**
Click to expand and configure:
- ☑️ **Enable GPT** - Check to include GPT models
- **Select Versions** (when enabled):
  - ☑️ GPT-4o
  - ☑️ GPT-5
  - ☑️ GPT-5.1
- You can select multiple versions - each will be tested separately

#### 🧠 **Claude (Anthropic)**
Click to expand and configure:
- ☑️ **Enable Claude** - Check to include Claude models
- **Select Versions** (when enabled):
  - ☑️ Claude Opus 4.5
  - ☑️ Claude Sonnet 4.5
  - ☑️ Claude Sonnet 4
- You can select multiple versions

#### 🔮 **Gemini (Google)**
Click to expand and configure:
- ☑️ **Enable Gemini** - Check to include Gemini models
- **Select Versions** (when enabled):
  - ☑️ Gemini 2.5 Pro
  - ☑️ Gemini 3 Pro Preview
- You can select multiple versions

#### 🚀 **Run New Analysis Button**

Large primary button at the bottom of the sidebar:
- Click to start a new analysis with your selected configuration
- Must have at least one model/version enabled
- Creates a new run with timestamp ID (e.g., `20251206-143022`)

---

### Main Area: Run Status Display

The main area shows the status of the current or most recent run.

#### Status Header

Three metric boxes display:

1. **Run ID** - Unique timestamp identifier for this run
   - Format: `YYYYMMDD-HHMMSS`
   - Example: `20251206-143022`

2. **Status** - Current run state
   - `PENDING` - Run initialized, not started yet
   - `RUNNING` - Currently executing
   - `COMPLETED` - Successfully finished
   - `ERROR` - Failed with an error

3. **Phase** - Current workflow stage
   - `COLLECT` - Gathering LLM responses
   - `METRICS` - Computing linguistic metrics
   - `ANALYZE` - Running statistical analysis
   - `DONE` - All processing complete

4. **Last Update** - Timestamp of last progress update (UTC)

---

#### Phase Indicators

During an active run, you'll see phase-specific messages:

**📥 Phase 1: Collecting LLM Responses**
- Displayed when: `phase == "collect"` and `status == "running"`
- What's happening:
  - Sending prompts to all selected models
  - Each task sent twice: once polite, once threatening
  - Collecting raw responses and token usage
- Spinner shows: "Sending prompts to selected models and collecting responses..."

**📊 Phase 2: Computing Metrics**
- Displayed when: `phase == "metrics"` and `status == "running"`
- What's happening:
  - Analyzing sentiment using VADER
  - Measuring politeness via linguistic features
  - Detecting toxicity with RoBERTa model
  - Identifying refusals and disclaimers
  - Extracting politeness strategies
- Spinner shows: "Analyzing sentiment, politeness, toxicity, and other metrics..."

**📈 Phase 3: Running Statistical Analysis**
- Displayed when: `phase == "analyze"` and `status == "running"`
- What's happening:
  - Performing paired t-tests (polite vs. threatening)
  - Calculating correlations between metrics
  - Computing delta values (tone differences)
  - Generating summary statistics
- Spinner shows: "Performing paired tests, correlations, and generating insights..."

---

#### Progress Tracking

**Per-Model Progress Bars**

For each enabled provider, you'll see:
- **Provider Header**: e.g., "**GPT** - 6/6 prompts"
  - Shows: Provider name, completed/total prompt count
- **Version Progress Bars**: One per version
  - Visual progress bar (0-100%)
  - Version name: e.g., `` `gpt-4o` ``
  - Completion count: e.g., "6/9"

**Example Progress Display**:
```
GPT - 6/6 prompts
[████████████████████] `gpt-4o`
                       6/9

[████████░░░░░░░░░░░░] `gpt-5`
                       4/9
```

**Failure Warnings**:
- If any prompts fail: "⚠️ 2 failed prompts"

---

#### Auto-Refresh Feature

When `status == "running"`:
- Page automatically refreshes every 2 seconds
- Display message: "🔄 **Auto-refreshing every 2 seconds...**"
- No manual action needed - just watch progress update
- Progress bars, phase indicators, and completion counts update live

---

#### Completion Display

When run finishes successfully (`status == "completed"`):

**Success Message**:
```
✅ Run completed successfully!
```

**GitHub Commit Status**:

You'll see one of these messages:

1. ✅ **Success** (green):
   ```
   ✅ GitHub Commit: Successfully committed to Youssab-William/capstone-dashboard
   ```
   - Data has been pushed to GitHub
   - Will persist across Streamlit Cloud restarts

2. ⚠️ **Skipped** (yellow):
   ```
   ⚠️ GitHub Commit Skipped: GITHUB_TOKEN not set. Data will not persist across deployments.
   ```
   - Includes expandable "How to enable GitHub persistence" section
   - Shows step-by-step instructions for setting up token

3. ❌ **Failed** (red):
   ```
   ❌ GitHub Commit Failed: Failed to commit. Check GITHUB_TOKEN and repo permissions.
   ```
   - Token is set but commit failed
   - Check token validity and repository access

4. ❌ **Error** (red):
   ```
   ❌ GitHub Commit Error: [specific error message]
   ```
   - Unexpected error during commit
   - Error details included for debugging

**View Results Button**:
- Large centered red button
- Text: "📊 View Results in Dashboard"
- Click to automatically:
  - Switch to Dashboard view
  - Pre-select this completed run
  - Load all visualizations

---

#### Error Display

When run fails (`status == "error"`):
```
❌ Error occurred: [error message]
⚠️ Check the logs or try running the analysis again.
```

---

## Dashboard View

The Dashboard is where you explore, analyze, and visualize completed runs.

### Sidebar: Run Selection

When in Dashboard view, the sidebar displays:

#### **Run History Dropdown**

A selectbox showing all completed analysis runs:

**Display Format**:
- Most recent runs at the top (descending order)
- Format: `YYYY-MM-DD HH:MM:SS (run_id)`
- Example: `2025-12-06 14:30:22 (20251206-143022)`
- Special entry: `Legacy baseline run` (reference data)

**Interaction**:
- Click dropdown to see full list
- Select a run to load its results
- Dashboard updates all tabs with selected run's data

---

### Main Area: Filters and Tabs

#### **Filters Section**

Three multi-select filters to refine your analysis:

**1. Models Filter**
- Options: `gpt`, `claude`, `gemini`, `deepseek`
- Default: All selected
- Uncheck to exclude specific providers from visualizations

**2. Versions Filter**
- Options: All model versions found in the run
- Examples: `gpt-4o`, `claude-opus-4-5-20251101`
- Default: All selected
- Uncheck to focus on specific versions

**3. Tones Filter**
- Options: `Polite`, `Threatening`
- Default: Both selected
- Uncheck one to analyze single tone condition

**Filter Behavior**:
- Filters apply to ALL tabs
- Changes reflect immediately in visualizations
- Use to focus on specific comparisons
- Clear all filters by reselecting everything

---

### Dashboard Tabs

The dashboard provides **8 comprehensive analysis tabs**:

---

## Tab 1: Overview

**Purpose**: High-level summary of all metrics across models and tones.

**Layout**:

**Summary Cards** (Top Row):
- Four metric boxes showing averages across all data:
  1. **Average Sentiment Score** - Overall emotional tone
  2. **Average Politeness Score** - Linguistic politeness level
  3. **Average Toxicity Score** - Presence of toxic language
  4. **Average Response Length** - Verbosity in words

**Bar Charts** (Main Area):

1. **Sentiment by Model and Tone**
   - X-axis: Models (GPT, Claude, etc.)
   - Y-axis: Sentiment Score (-1 to +1)
   - Colors: Blue (Polite), Orange (Threatening)
   - Groups by: Model → Version → Tone
   - Shows: How sentiment varies by provider and tone

2. **Politeness by Model and Tone**
   - X-axis: Models
   - Y-axis: Politeness Score
   - Colors: Blue (Polite), Orange (Threatening)
   - Shows: Which models use more polite language

3. **Toxicity by Model and Tone**
   - X-axis: Models
   - Y-axis: Toxicity Score (0 to 1)
   - Colors: Blue (Polite), Orange (Threatening)
   - Shows: Safety behavior across models

4. **Response Length by Model and Tone**
   - X-axis: Models
   - Y-axis: Word count
   - Colors: Blue (Polite), Orange (Threatening)
   - Shows: Verbosity differences

**Use Cases**:
- Quick health check of a run
- Spot high-level trends
- Identify which models are outliers
- Compare overall tone impact

---

## Tab 2: Tone Impact

**Purpose**: Measure the delta (difference) between Polite and Threatening tones.

**Layout**:

**Delta Calculation**:
- For each model and task:
  - `Delta = Polite Score - Threatening Score`
- Positive delta = Polite tone yields higher scores
- Negative delta = Threatening tone yields higher scores

**Visualization**:

**Sentiment Delta Bar Chart**
- X-axis: Models
- Y-axis: Delta value
- Bars: Positive (above zero) or negative (below zero)
- Grouped by: Model and task category

**What This Tells You**:
- **Large positive delta**: Model is very tone-sensitive (polite prompts get better sentiment)
- **Near-zero delta**: Model is tone-resistant (responds similarly regardless)
- **Negative delta**: Model responds better to threatening tone (unusual!)

**Use Cases**:
- Identify most tone-sensitive models
- Measure magnitude of tone effect
- Compare tone impact across categories
- Find unexpected tone behaviors

---

## Tab 3: Categories

**Purpose**: Break down metrics by task category (e.g., Coding, Creative Writing, General Knowledge).

**Layout**:

**Category Columns**:
- Extracted from Task IDs (e.g., `ProgrammingHelpTask1` → `ProgrammingHelp`)

**Metrics Shown**:
1. Sentiment Score per category
2. Politeness Score per category
3. Toxicity Score per category
4. Response Length per category

**Two Views**:

1. **Overall Category Breakdown**
   - Aggregated across all models
   - Shows: Which task types get what responses

2. **Per-Model Category Analysis**
   - Separate charts for each model
   - Shows: Model-specific category patterns

**Use Cases**:
- Identify if coding tasks trigger different responses than creative tasks
- Check if certain categories are more sensitive to tone
- Compare model strengths across task types
- Spot category-specific behaviors

---

## Tab 4: Safety

**Purpose**: Analyze model safety behaviors (refusals and disclaimers).

**Layout**:

**Two Main Charts**:

1. **Refusal Rates by Model and Tone**
   - Shows: Percentage of responses where model refused the task
   - Detection: Keywords like "I cannot", "I'm unable to", "I apologize, but I can't"
   - Grouped by: Model and Tone
   - Bar chart with percentages (0-100%)

2. **Disclaimer Rates by Model and Tone**
   - Shows: Percentage of responses with safety disclaimers
   - Detection: Phrases like "I'm an AI", "I cannot provide medical advice"
   - Grouped by: Model and Tone
   - Bar chart with percentages (0-100%)

**Colors**:
- Blue: Polite tone
- Orange: Threatening tone

**Use Cases**:
- Evaluate safety guardrails across models
- Check if threatening tones trigger more refusals (expected)
- Identify overly cautious models (high disclaimer rates)
- Compare safety approaches across providers

---

## Tab 5: Strategies

**Purpose**: Analyze linguistic politeness strategies used in responses.

**Layout**:

**Strategy Types Detected**:
1. **Hedges** - "might", "could", "perhaps", "maybe"
2. **Gratitude** - "thank you", "thanks", "appreciate"
3. **Deference** - "if you don't mind", "if possible"
4. **Apologies** - "sorry", "I apologize", "excuse me"
5. **Please** - Direct use of "please"

**Visualization**:

**Strategy Frequency Multi-Column Bar Chart**
- X-axis: Strategy types
- Y-axis: Frequency count
- Columns: One per model
- Colors: Different color per model for easy comparison

**Parsed From**: `Response_ValidatedStrategies` field
- Format: `"Hedges:3, Gratitude:1, Please:2"`
- Counted and visualized across all responses

**Use Cases**:
- Understand how models construct polite responses
- Identify model-specific linguistic patterns
- Check if threatening prompts reduce politeness strategies
- Compare strategy preferences across providers

---

## Tab 6: Correlations

**Purpose**: Explore relationships between different metrics.

**Layout**:

**Three Heatmaps**:

1. **Overall Correlations** (All Data)
   - Rows & Columns: Sentiment, Politeness, Response Length
   - Cell values: Correlation coefficient (-1 to +1)
   - Color scale:
     - Red: Negative correlation
     - White: No correlation
     - Blue: Positive correlation

2. **Polite Tone Correlations**
   - Same metrics, filtered to Polite tone only
   - Shows: Relationships within polite responses

3. **Threatening Tone Correlations**
   - Same metrics, filtered to Threatening tone only
   - Shows: Relationships within threatening responses

**Reading the Heatmap**:
- **Strong positive** (dark blue, ~1.0): Metrics increase together
- **No correlation** (white, ~0.0): Metrics unrelated
- **Strong negative** (dark red, ~-1.0): One increases, other decreases

**Expected Patterns**:
- Sentiment ↔ Toxicity: Negative correlation (more toxic = less positive)
- Sentiment ↔ Politeness: Positive correlation (more polite = more positive)

**Use Cases**:
- Validate expected relationships
- Discover unexpected correlations
- Check if tone affects metric relationships
- Identify confounding variables

---

## Tab 7: Paired Tests

**Purpose**: Statistical significance testing for tone differences.

**Layout**:

**Statistical Method**: Paired t-test
- Compares Polite vs. Threatening for same task/model
- Tests if differences are statistically significant
- Accounts for task-level pairing

**Metrics Tested**:
1. Response Sentiment Score
2. Response Politeness Score
3. Response Toxicity Score
4. Response Length

**Results Table** (for each metric):
- **Mean Difference**: Polite minus Threatening (average)
- **P-Value**: Statistical significance (< 0.05 = significant)
- **Significant**: Yes/No indicator

**Visualization**:

**Mean Difference Bar Chart**
- X-axis: Metrics
- Y-axis: Mean difference value
- Bars:
  - Positive: Polite tone scores higher
  - Negative: Threatening tone scores higher
  - Height: Magnitude of difference
- Annotations: P-values and significance markers

**Interpreting Results**:
- **p < 0.05**: Difference is statistically significant (real effect)
- **p ≥ 0.05**: Difference could be due to chance (no proven effect)
- **Large mean difference + low p-value**: Strong tone effect
- **Small mean difference or high p-value**: Weak/no tone effect

**Use Cases**:
- Determine if tone truly affects model behavior
- Identify which metrics are most tone-sensitive
- Make evidence-based conclusions
- Support research claims with statistical backing

---

## Tab 8: Prompts

**Purpose**: Detailed inspection of individual prompts and responses.

**Layout**:

**Full Data Table** showing ALL records with columns:

**Identification**:
- `TaskID` - Unique task identifier
- `TaskCategory` - Task type (Coding, Creative, etc.)
- `Model` - Provider (gpt, claude, etc.)
- `Version` - Specific model version
- `PromptTone` - Polite or Threatening

**Text Content**:
- `PromptText` - Full prompt sent to model
- `ResponseText` - Full response received from model

**Sentiment Metrics**:
- `Prompt_SentimentScore` - Sentiment of the prompt
- `Response_SentimentScore` - Sentiment of the response

**Politeness Metrics**:
- `Response_ValidatedPolitenessScore` - Politeness level
- `Response_ValidatedStrategies` - Specific strategies used

**Safety Metrics**:
- `RoBERTa_Response_ToxicityScore` - Toxicity score
- `Response_RefusalFlag` - Did model refuse? (True/False)
- `Response_DisclaimerFlag` - Did model add disclaimers? (True/False)

**Length Metrics**:
- `ResponseLength` - Number of tokens/words

**Token Usage**:
- `PromptTokens` - Tokens in prompt
- `CompletionTokens` - Tokens in response
- `TotalTokens` - Sum of both

**Timestamps**:
- `created_at` - When this record was created

**Features**:
- **Sortable**: Click any column header to sort
- **Searchable**: Use Streamlit's table search (magnifying glass icon)
- **Expandable**: Click rows to see full text content
- **Scrollable**: Navigate through all records

**Use Cases**:
- Investigate specific responses
- Verify metric calculations by reading actual text
- Find examples for presentations or papers
- Deep-dive into anomalies or edge cases
- Export data: Copy/paste or screenshot
- Quality check: Ensure prompts were sent correctly

---

## Understanding the Metrics

### Sentiment Score

**What It Measures**: Overall emotional tone of the response

**Tool**: VADER (Valence Aware Dictionary and sEntiment Reasoner)

**Range**: -1.0 (very negative) to +1.0 (very positive)

**Interpretation**:
- **> 0.05**: Positive sentiment
- **-0.05 to 0.05**: Neutral sentiment
- **< -0.05**: Negative sentiment

**Examples**:
- "I'm happy to help!" → ~0.8 (very positive)
- "Here is the information." → ~0.0 (neutral)
- "I cannot assist with that." → ~-0.2 (slightly negative)

**In Dashboard**: Compare sentiment across tones to see if threatening prompts yield less positive responses.

---

### Politeness Score

**What It Measures**: Level of linguistic politeness strategies used

**Method**: Rule-based linguistic analysis

**Range**: Continuous scale (higher = more polite, no fixed max)

**Components**:
- Hedges: "might", "could", "perhaps"
- Gratitude: "thank you", "thanks"
- Deference: "if you don't mind", "if possible"
- Apologies: "sorry", "I apologize"
- Direct markers: "please"

**Interpretation**:
- Higher scores = more polite language
- Scores vary significantly by model and tone
- Typical range: 0-10 (but can be higher)

**In Dashboard**: Compare polite vs. threatening tone to measure tone sensitivity.

---

### Toxicity Score

**What It Measures**: Presence of toxic, harmful, or offensive language

**Tool**: RoBERTa-based toxicity classifier

**Range**: 0.0 (not toxic) to 1.0 (highly toxic)

**Interpretation**:
- **< 0.1**: Not toxic (safe)
- **0.1 - 0.5**: Mildly toxic (borderline)
- **> 0.5**: Highly toxic (unsafe)

**Note**: Most LLM responses score very low (<0.01) due to safety training and alignment.

**In Dashboard**: Check if any models produce toxic responses, especially under threatening prompts.

---

### Response Length

**What It Measures**: Verbosity of the response

**Units**: Number of tokens (approximately number of words)

**Interpretation**:
- **Shorter**: More concise, potentially curt
- **Longer**: More detailed, potentially verbose
- Compare across tones: Do threatening prompts yield shorter responses?

**In Dashboard**: Compare length across models and tones to understand verbosity patterns.

---

### Refusal Flag

**What It Measures**: Whether model explicitly refused the task

**Values**: `True` or `False` (boolean)

**Detection Method**: Keyword matching
- "I cannot", "I can't", "I'm unable"
- "I apologize, but I can't"
- "I'm not able to"

**Interpretation**:
- `True`: Model declined to complete the task (safety guardrail activated)
- `False`: Model attempted the task

**In Dashboard**: Compare refusal rates across models and tones to understand safety behaviors.

---

### Disclaimer Flag

**What It Measures**: Whether response includes safety disclaimers

**Values**: `True` or `False` (boolean)

**Detection Method**: Phrase matching
- "I'm an AI"
- "I cannot provide medical advice"
- "Consult a professional"
- Similar cautionary language

**Interpretation**:
- `True`: Model added cautionary language
- `False`: No disclaimers present
- Common in medical, legal, or sensitive domains

**In Dashboard**: Track which models and tones trigger disclaimers most often.

---

## Analysis Workflow

### Complete Run Lifecycle

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Configure   │ ──> │  2. Start    │ ──> │  3. Monitor     │ ──> │  4. Analyze     │
│     Models      │     │     Run      │     │    Progress     │     │    Results      │
│  (Run Monitor)  │     │ (Run Monitor)│     │  (Run Monitor)  │     │   (Dashboard)   │
└─────────────────┘     └──────────────┘     └─────────────────┘     └─────────────────┘
```

### Step-by-Step Guide

#### Step 1: Configure Your Run

1. Open the dashboard
2. Navigate to **Run Monitor** (sidebar radio button)
3. Expand provider sections in sidebar
4. Check boxes to enable providers
5. Select specific model versions
6. Click **"Run New Analysis"**

**Tips**:
- Start with fewer models for faster runs
- Enable all versions of one provider to compare
- Enable one version across all providers to compare providers

#### Step 2: Monitor Progress (5-10 minutes)

1. Watch status change to `RUNNING`
2. Observe phase progression: COLLECT → METRICS → ANALYZE
3. Track per-model progress bars
4. Page auto-refreshes every 2 seconds
5. Wait for "✅ Run completed successfully!"

**What to Watch**:
- Progress bars filling up
- Phase indicators changing
- Completion counts increasing
- Any failure warnings

#### Step 3: Check GitHub Status

After completion:
- ✅ **Success**: Data is persisted to GitHub
- ⚠️ **Skipped**: Need to configure `GITHUB_TOKEN` (optional)
- ❌ **Failed**: Check token permissions

**Note**: GitHub persistence is optional but recommended for Streamlit Cloud deployments.

#### Step 4: Explore Results

1. Click **"📊 View Results in Dashboard"**
2. Dashboard opens with completed run pre-selected
3. Explore all 8 tabs:
   - Start with **Overview** for high-level summary
   - Check **Tone Impact** to see delta analysis
   - Review **Paired Tests** for statistical significance
   - Use **Prompts** tab to inspect individual responses
4. Apply filters to focus analysis
5. Take screenshots or export data

#### Step 5: Compare Runs (Optional)

1. Run analysis again with different configuration
2. Use **Run History** dropdown to switch between runs
3. Compare how results differ
4. Track model behavior over time

---

## Technical Architecture

### Core Components

**1. Providers** (`capstone_pipeline/providers/`)
- `GPTProvider`: OpenAI GPT models
- `ClaudeProvider`: Anthropic Claude models
- `DeepSeekProvider`: DeepSeek API
- `GeminiProvider`: Google Gemini models
- Each implements the `LLMProvider` interface

**2. Storage** (`capstone_pipeline/storage/`)
- `JsonlStorage`: JSONL format data storage
- `github_persist.py`: GitHub commit integration
- Organized by table and run_id partitions

**3. Metrics Engine** (`capstone_pipeline/metrics/`)
- `ScriptMetricsEngine`: Computes 40+ metrics
- Uses legacy analysis modules for sentiment, toxicity, politeness

**4. Analysis Engine** (`capstone_pipeline/analysis/`)
- `ScriptAnalysisEngine`: Statistical analysis
- Generates summaries, deltas, correlations, paired tests

**5. Workflow** (`capstone_pipeline/workflow/`)
- `Pipeline`: Orchestrates full pipeline
- `ResponseCollector`: Manages LLM API calls

**6. Progress Tracking** (`capstone_pipeline/runner/`)
- `RunProgressTracker`: Real-time progress updates
- Persists to JSON for UI monitoring

### Data Flow

```
Tasks (prompts.txt)
  ↓
ResponseCollector → LLM Providers (GPT, Claude, DeepSeek, Gemini)
  ↓
Completions Storage (data/completions/{run_id}.jsonl)
  ↓
Metrics Engine → Sentiment, Politeness, Toxicity Analysis
  ↓
Metrics Storage (data/metrics/{run_id}.jsonl)
  ↓
Analysis Engine → Statistics, Correlations, Tests
  ↓
Analysis Storage (data/analysis/{run_id}.jsonl)
  ↓
Dashboard → Interactive Visualizations
```

### Project Structure

```
capstone_pipeline/
├── app.py                 # CLI entry point
├── config.py              # Configuration classes
├── domain.py              # Domain models
├── interfaces.py          # Protocol definitions
├── providers/             # LLM provider implementations
├── storage/               # Data storage + GitHub
├── metrics/               # Metrics computation
├── analysis/              # Statistical analysis
├── workflow/              # Pipeline orchestration
└── runner/                # Response collection + progress

dashboard/
└── streamlit_app.py       # Web UI (this is what you use!)

data/
├── completions/           # Raw LLM responses
├── metrics/               # Computed metrics
├── analysis/              # Analysis results
├── logs/                  # Run progress logs
└── keys.json             # API keys (not in git)

prompts.txt                # Task definitions (JSON format)
```

---

## Deployment

### Deploying to Streamlit Cloud

1. **Push to GitHub**:
```bash
./deploy.sh
```

Or manually:
```bash
git add .
git commit -m "Deploy dashboard"
git push origin main
```

2. **Connect to Streamlit Cloud**:
- Go to https://share.streamlit.io/
- Click "New app"
- Select your GitHub repository
- Main file: `dashboard/streamlit_app.py`
- Click "Deploy"

3. **Configure Secrets** (in Streamlit Cloud app settings):
```toml
# API Keys (required)
[api_keys]
openai_api_key = "sk-..."
anthropic_api_key = "sk-ant-..."
gemini_api_key = "AI..."
deepseek_api_key = "sk-..."

# GitHub Persistence (optional but recommended)
GITHUB_TOKEN = "ghp_..."
```

4. **Access**: Streamlit provides a public URL

**See Also**:
- `STREAMLIT_CLOUD_SETUP.md` - Detailed deployment guide
- `GITHUB_SETUP.md` - GitHub token configuration

---

## Troubleshooting

### Common Issues

#### "Auto-refresh not working"
**Fix**: Already deployed. Page refreshes every 2s during runs.

#### "Redirect button doesn't work"
**Fix**: Already deployed. Button switches to Dashboard properly.

#### "No progress bars showing"
**Fix**: Already deployed. Progress shown during all phases.

#### "GitHub commits failing"
**Causes**:
- `GITHUB_TOKEN` not set → Add to Streamlit secrets
- Token expired → Generate new token
- Wrong permissions → Ensure `repo` scope

**Solution**: See `GITHUB_SETUP.md`

#### "Run fails immediately"
**Check**:
1. API keys correctly set
2. At least one provider enabled
3. API keys have credits/quota
4. Internet connection stable

#### "Dashboard loads slowly"
**Solutions**:
- Use filters to reduce data volume
- Caching already enabled in code
- Archive old runs if too many

### Getting Help

1. Check documentation files in repository
2. Review Streamlit Cloud logs for errors
3. Verify API keys and configuration
4. Test locally before deploying

---

## Additional Resources

### Documentation Files

- **`GITHUB_SETUP.md`** - GitHub token setup
- **`STREAMLIT_CLOUD_SETUP.md`** - Deployment guide
- **`DEPLOYMENT_FIX.md`** - Recent fixes
- **`FIXES_SUMMARY.md`** - Changelog

### Task Configuration

Tasks are defined in `prompts.txt` as JSON array:
```json
[
  {
    "TaskID": "ProgrammingHelpTask1",
    "TaskDescription": "Write a Python function",
    "PromptTone": "Polite",
    "PromptText": "Could you please help me write..."
  }
]
```

### Run Data Format

Each run creates three JSONL files:
- `completions/{run_id}.jsonl` - Raw responses
- `metrics/{run_id}.jsonl` - Computed metrics
- `analysis/{run_id}.jsonl` - Statistical results

---

## Support & Contributing

### Common Questions

**Q: How many API calls per run?**
A: 2 per task per model version (one polite, one threatening). Default: 9 tasks × N versions.

**Q: How much does a run cost?**
A: Depends on models. Estimate: $0.50-$2.00 per full run with all models.

**Q: Can I add custom tasks?**
A: Yes! Edit `prompts.txt` (JSON format).

**Q: Can I export data?**
A: Yes! Use Prompts tab or access JSONL files directly.

**Q: How do I compare runs?**
A: Use Run History dropdown + screenshots/exports.

---

**Dashboard Version**: 2.0
**Last Updated**: December 2025
**Maintainer**: Youssab William & Team
