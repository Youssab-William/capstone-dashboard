# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** The `detoxify` package may take a while to install as it downloads ML models.

## Step 2: Verify API Keys

Your API keys are already configured in `data/keys.json`. The pipeline will automatically load them when you use the `--keys_file` flag.

## Step 3: Run the Pipeline

### Option A: Full Pipeline (Recommended for first run)

This will:
1. Collect responses from all LLM providers (GPT, Claude, DeepSeek, Gemini)
2. Compute metrics (sentiment, toxicity, politeness, etc.)
3. Perform statistical analysis

```bash
python -m capstone_pipeline.app run --keys_file data/keys.json --prompts_file prompts.txt
```

**Expected output:**
- A new `run_id` will be generated (format: `YYYYMMDD-HHMMSS`)
- Responses saved to `data/completions/{run_id}.jsonl`
- Metrics saved to `data/metrics/{run_id}.jsonl`
- Analysis saved to `data/analysis/{run_id}.jsonl`

### Option B: Step-by-Step (For debugging)

```bash
# Step 1: Collect responses only
python -m capstone_pipeline.app collect --keys_file data/keys.json --prompts_file prompts.txt

# Note the run_id from the output, then:
python -m capstone_pipeline.app metrics --run_id YOUR_RUN_ID
python -m capstone_pipeline.app analyze --run_id YOUR_RUN_ID
```

## Step 4: View Results

### Using the Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Using Command Line

View the JSONL files directly:
```bash
# View completions
cat data/completions/{run_id}.jsonl | head -1 | python -m json.tool

# View metrics
cat data/metrics/{run_id}.jsonl | head -1 | python -m json.tool

# View analysis
cat data/analysis/{run_id}.jsonl | head -1 | python -m json.tool
```

## Troubleshooting

### Issue: "Module not found"
**Solution:** Make sure you're in the project root directory and dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: "API key error"
**Solution:** Check that `data/keys.json` exists and contains valid API keys. You can also set environment variables:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
# etc.
```

### Issue: "detoxify model download fails"
**Solution:** The first run of detoxify downloads models. If it fails, try:
```bash
pip install --upgrade detoxify
python -c "from detoxify import Detoxify; Detoxify('unbiased')"
```

### Issue: "No completions found"
**Solution:** The pipeline skips existing completions. To force re-collection, delete the specific run_id file:
```bash
rm data/completions/{run_id}.jsonl
```

## Understanding the Output

- **Completions**: Raw responses from LLMs with metadata
- **Metrics**: Computed scores (sentiment, toxicity, politeness, refusal flags)
- **Analysis**: Statistical summaries, deltas, correlations, and paired tests

## Next Steps

- Modify `prompts.txt` to add your own tasks
- Customize providers in `capstone_pipeline/providers/`
- Extend metrics in `capstone_pipeline/metrics/`
- Add visualizations in `dashboard/streamlit_app.py`

