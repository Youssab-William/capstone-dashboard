import os
import sys
import json
from pathlib import Path
import threading
import time
from datetime import datetime, timezone

# Add project root to Python path for imports (needed for deployment)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import altair as alt

from capstone_pipeline.workflow.pipeline import Pipeline
from capstone_pipeline.storage.repository import JsonlStorage
from capstone_pipeline.metrics.metrics_engine import ScriptMetricsEngine
from capstone_pipeline.analysis.analysis_engine import ScriptAnalysisEngine
from capstone_pipeline.app import parse_tasks_file
from capstone_pipeline.providers.deepseek import DeepSeekProvider
from capstone_pipeline.providers.gpt import GPTProvider
from capstone_pipeline.providers.claude import ClaudeProvider
from capstone_pipeline.providers.gemini import GeminiProvider
from capstone_pipeline.config import PROVIDER_MODELS
from capstone_pipeline.runner.progress import RunProgressTracker


def list_run_ids(data_dir: str) -> list:
    metrics_dir = Path(data_dir) / "metrics"
    if not metrics_dir.exists():
        return []
    return [p.stem for p in metrics_dir.glob("*.jsonl")]


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_run_progress(data_dir: str, run_id: str) -> dict:
    """
    Read progress metadata for a given run_id, if available.
    """
    path = Path(data_dir) / "logs" / f"run_{run_id}.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def render_run_monitor(data_dir: str, prompts_file: str) -> None:
    """
    Run Monitor page showing current/last run status.
    Auto-refreshes every 2 seconds when a run is active.
    """
    st.subheader("Run Monitor")
    
    # Auto-refresh every 2 seconds if there's an active run
    active_run_id = st.session_state.get("active_run_id") or ""
    if active_run_id:
        prog = read_run_progress(data_dir, active_run_id)
        if prog and prog.get("status") == "running":
            # Auto-refresh using JavaScript
            st.markdown("""
                <script>
                    setTimeout(function(){
                        window.location.reload(1);
                    }, 2000);
                </script>
            """, unsafe_allow_html=True)
            st.info("🔄 Auto-refreshing every 2 seconds...")
    
    # Try to get active run or most recent
    if not active_run_id:
        from glob import glob
        logs = sorted(glob(f"{data_dir}/logs/run_*.json"))
        if logs:
            latest = max(logs, key=os.path.getmtime)
            active_run_id = Path(latest).stem.replace("run_", "")
            st.session_state["active_run_id"] = active_run_id
    
    if not active_run_id:
        st.info("No run detected. Configure models in the sidebar and click 'Run New Analysis'.")
        return
    
    prog = read_run_progress(data_dir, active_run_id)
    if not prog:
        st.info(f"No progress metadata found for run_id={active_run_id}.")
        return
    
    # Status display
    status = prog.get("status", "unknown")
    phase = prog.get("phase", "")
    
    cols = st.columns(3)
    cols[0].metric("Run ID", active_run_id)
    cols[1].metric("Status", status)
    cols[2].metric("Phase", phase)
    
    st.write(f"Last update (UTC): `{prog.get('updated_at', '')[:19]}`")
    
    # Loading screens based on phase
    if phase == "metrics":
        st.info("📊 **Metrics are being collected...**")
        st.spinner("Computing metrics for all responses")
    elif phase == "analyze":
        st.info("📈 **Analysis is being calculated...**")
        st.spinner("Running statistical analysis")
    elif status == "completed":
        st.success("✅ **Run completed!** Redirecting to Dashboard...")
        if st.button("View Results in Dashboard Now", type="primary"):
            st.query_params["view"] = "Dashboard"
            st.query_params["run"] = active_run_id
            st.rerun()
    elif status == "error":
        st.error(f"❌ **Error**: {prog.get('error', 'Unknown error')}")
    
    # Progress display
    selection = prog.get("selection", {}) or {}
    per_model = prog.get("per_model", {}) or {}
    tasks = parse_tasks_file(prompts_file)
    num_tasks = len(tasks)
    
    if selection:
        st.markdown("#### Per-model / per-version progress")
        for provider_key, cfg in selection.items():
            if not cfg.get("enabled"):
                continue
            versions = cfg.get("versions") or []
            if not versions:
                continue
            stats = per_model.get(provider_key, {}) or {}
            model_total = max(int(stats.get("total", 0)), 0)
            model_done = max(int(stats.get("completed", 0)) + int(stats.get("failed", 0)), 0)
            model_frac = float(model_done) / model_total if model_total > 0 else 0.0
            
            st.markdown(f"**{provider_key}**")
            for ver in versions:
                expected = num_tasks
                actual = int(round(model_frac * expected))
                actual = min(actual, expected)
                frac = float(actual) / expected if expected > 0 else 0.0
                st.write(f"- `{ver}` – {actual}/{expected}")
                st.progress(min(frac, 1.0))
    
    # Manual refresh button
    if status == "running":
        if st.button("🔄 Refresh Progress", type="primary"):
            st.rerun()
    
    # Auto-redirect when completed (but don't break sidebar)
    if status == "completed":
        if "redirected" not in st.session_state or st.session_state.get("redirected_run_id") != active_run_id:
            st.session_state["redirected"] = True
            st.session_state["redirected_run_id"] = active_run_id
            st.session_state["selected_run_id"] = active_run_id
            # Use JavaScript to redirect after showing message
            st.markdown("""
                <script>
                    setTimeout(function(){
                        window.location.href = window.location.pathname + "?view=Dashboard&run=" + arguments[0];
                    }, 2000);
                </script>
            """, unsafe_allow_html=True)
        st.markdown("### Current Run Status")
        
        # Try to get active run or most recent
        if not active_run_id:
            from glob import glob
            logs = sorted(glob(f"{data_dir}/logs/run_*.json"))
            if logs:
                latest = max(logs, key=os.path.getmtime)
                active_run_id = Path(latest).stem.replace("run_", "")
                st.session_state["active_run_id"] = active_run_id
        
        if not active_run_id:
            st.info("No run detected. Configure models above and click 'Run New Analysis'.")
            return
        
        prog = read_run_progress(data_dir, active_run_id)
        if not prog:
            st.info(f"No progress metadata found for run_id={active_run_id}.")
            return
        
        # Status display
        status = prog.get("status", "unknown")
        phase = prog.get("phase", "")
        
        cols = st.columns(3)
        cols[0].metric("Run ID", active_run_id)
        cols[1].metric("Status", status)
        cols[2].metric("Phase", phase)
        
        st.write(f"Last update (UTC): `{prog.get('updated_at', '')[:19]}`")
        
        # Loading screens based on phase
        if phase == "metrics":
            st.info("📊 **Metrics are being collected...**")
            st.spinner("Computing metrics for all responses")
        elif phase == "analyze":
            st.info("📈 **Analysis is being calculated...**")
            st.spinner("Running statistical analysis")
        elif status == "completed":
            st.success("✅ **Run completed!**")
            if st.button("View Results in Dashboard", type="primary"):
                st.session_state["selected_run_id"] = active_run_id
                st.session_state["view"] = "Dashboard"
                st.rerun()
        elif status == "error":
            st.error(f"❌ **Error**: {prog.get('error', 'Unknown error')}")
        
        # Progress display
        selection = prog.get("selection", {}) or {}
        per_model = prog.get("per_model", {}) or {}
        tasks = parse_tasks_file(prompts_file)
        num_tasks = len(tasks)
        
        if selection:
            st.markdown("#### Per-model / per-version progress")
            for provider_key, cfg in selection.items():
                if not cfg.get("enabled"):
                    continue
                versions = cfg.get("versions") or []
                if not versions:
                    continue
                stats = per_model.get(provider_key, {}) or {}
                model_total = max(int(stats.get("total", 0)), 0)
                model_done = max(int(stats.get("completed", 0)) + int(stats.get("failed", 0)), 0)
                model_frac = float(model_done) / model_total if model_total > 0 else 0.0
                
                st.markdown(f"**{provider_key}**")
                for ver in versions:
                    expected = num_tasks
                    actual = int(round(model_frac * expected))
                    actual = min(actual, expected)
                    frac = float(actual) / expected if expected > 0 else 0.0
                    st.write(f"- `{ver}` – {actual}/{expected}")
                    st.progress(min(frac, 1.0))
        
        # Auto-redirect when completed
        if status == "completed" and "redirected" not in st.session_state:
            st.session_state["redirected"] = True
            st.session_state["selected_run_id"] = active_run_id
            time.sleep(2)  # Show completion message briefly
            st.session_state["view"] = "Dashboard"
            st.rerun()


def build_providers_from_selection(selection: dict) -> list:
    """
    Build provider instances based on user selections from the sidebar.

    selection structure:
    {
        "gpt": {"enabled": True, "versions": ["gpt-4o", "gpt-4.1"]},
        "claude": {...},
        "gemini": {...},
        "deepseek": {"enabled": True, "versions": ["deepseek-chat"]},
    }
    """
    providers = []
    if selection.get("deepseek", {}).get("enabled"):
        versions = selection["deepseek"].get("versions") or ["deepseek-chat"]
        providers.append(DeepSeekProvider(versions=versions))
    if selection.get("gpt", {}).get("enabled"):
        versions = selection["gpt"].get("versions") or [v.id for v in PROVIDER_MODELS["gpt"].versions]
        providers.append(GPTProvider(versions=versions))
    if selection.get("claude", {}).get("enabled"):
        versions = selection["claude"].get("versions") or [v.id for v in PROVIDER_MODELS["claude"].versions]
        providers.append(ClaudeProvider(versions=versions))
    if selection.get("gemini", {}).get("enabled"):
        versions = selection["gemini"].get("versions") or [v.id for v in PROVIDER_MODELS["gemini"].versions]
        providers.append(GeminiProvider(versions=versions))
    return providers


def start_background_run(data_dir: str, prompts_file: str, keys_file: str, run_id: str, provider_selection: dict) -> None:
    """
    Orchestrate a full pipeline run in a background thread.

    This function is intended to be launched via threading.Thread(target=...).
    """
    logger = st.logger.get_logger("streamlit_pipeline") if hasattr(st, "logger") else None
    try:
        # Load keys and configure environment
        # Priority: 1) keys_file (if exists), 2) environment variables (already set)
        if keys_file and os.path.exists(keys_file):
            with open(keys_file, "r", encoding="utf-8") as f:
                keys = json.load(f)

            def set_key(env_name: str, value: str) -> None:
                if value:
                    os.environ[env_name] = value

            set_key("DEEPSEEK_API_KEY", keys.get("deepseek_api_key") or keys.get("DEEPSEEK_API_KEY"))
            set_key("OPENAI_API_KEY", keys.get("openai_api_key") or keys.get("OPENAI_API_KEY"))
            set_key("ANTHROPIC_API_KEY", keys.get("anthropic_api_key") or keys.get("ANTHROPIC_API_KEY"))
            set_key("GEMINI_API_KEY", keys.get("gemini_api_key") or keys.get("GEMINI_API_KEY"))
            set_key("GOOGLE_API_KEY", keys.get("google_api_key") or keys.get("GOOGLE_API_KEY"))
        
        # Log which API keys are available (without exposing the actual keys)
        api_keys_status = {
            "OPENAI_API_KEY": "✓" if os.environ.get("OPENAI_API_KEY") else "✗",
            "ANTHROPIC_API_KEY": "✓" if os.environ.get("ANTHROPIC_API_KEY") else "✗",
            "DEEPSEEK_API_KEY": "✓" if os.environ.get("DEEPSEEK_API_KEY") else "✗",
            "GEMINI_API_KEY": "✓" if os.environ.get("GEMINI_API_KEY") else "✗",
            "GOOGLE_API_KEY": "✓" if os.environ.get("GOOGLE_API_KEY") else "✗",
        }
        if logger:
            logger.info(f"API keys status: {api_keys_status}")

        storage = JsonlStorage(data_dir)
        metrics = ScriptMetricsEngine()
        analysis = ScriptAnalysisEngine()
        pipeline = Pipeline(storage, metrics, analysis)
        tasks = parse_tasks_file(prompts_file)
        providers = build_providers_from_selection(provider_selection)
        tracker = RunProgressTracker(data_dir, run_id)
        # Persist the selection so we can later display accurate per-model /
        # per-version progress even after Streamlit is restarted.
        tracker.set_selection(provider_selection)
        if not providers:
            # Nothing to do; mark run as error so UI can surface it.
            tracker.set_status("error", error="No providers selected for this run.")
            return
        pipeline.run_all(tasks, providers, run_id=run_id, progress=tracker)
        
        # After successful run, commit data to GitHub for persistence
        try:
            from capstone_pipeline.storage.github_persist import commit_run_data_to_github
            github_token = os.environ.get("GITHUB_TOKEN")
            repo_owner = os.environ.get("GITHUB_REPO_OWNER", "Youssab-William")
            repo_name = os.environ.get("GITHUB_REPO_NAME", "capstone-dashboard")
            
            if github_token:
                success = commit_run_data_to_github(
                    data_dir=data_dir,
                    run_id=run_id,
                    github_token=github_token,
                    repo_owner=repo_owner,
                    repo_name=repo_name,
                )
                if success and logger:
                    logger.info(f"Successfully committed run {run_id} data to GitHub")
                elif logger:
                    logger.warning(f"Failed to commit run {run_id} data to GitHub (check GITHUB_TOKEN)")
            elif logger:
                logger.info(f"GITHUB_TOKEN not set, skipping GitHub commit for run {run_id}")
        except Exception as e:
            # Don't fail the run if GitHub commit fails
            if logger:
                logger.warning(f"Error committing to GitHub (non-fatal): {e}")
    except Exception as e:
        # If anything goes wrong, persist an error state.
        tracker = RunProgressTracker(data_dir, run_id)
        tracker.set_status("error", error=str(e))
        if logger is not None:
            logger.exception("background_pipeline_run_failed run_id=%s error=%s", run_id, e)


def main():
    st.set_page_config(page_title="LLM Tone Dashboard", layout="wide")
    st.title("LLM Tone Effects Dashboard")

    data_dir = "data"
    prompts_file = "prompts.txt"
    keys_file = "data/keys.json"

    # Choose view: main analytics dashboard vs. run monitor
    # Check URL params or session state for view selection
    query_params = st.query_params
    if "view" in query_params:
        view = query_params["view"]
        if view not in ["Dashboard", "Run Monitor"]:
            view = "Dashboard"
    elif "view" in st.session_state:
        view = st.session_state["view"]
        # Don't delete it, keep it for consistency
    else:
        view = st.sidebar.radio("View", ["Dashboard", "Run Monitor"], index=0)
    
    # Update session state with current view
    st.session_state["view"] = view
    
    # Initialize provider selection if not exists
    if "provider_selection" not in st.session_state:
        st.session_state["provider_selection"] = {
            "deepseek": {"enabled": True, "versions": ["deepseek-chat"]},
            "gpt": {
                "enabled": True,
                "versions": [v.id for v in PROVIDER_MODELS["gpt"].versions],
            },
            "claude": {
                "enabled": True,
                "versions": [v.id for v in PROVIDER_MODELS["claude"].versions],
            },
            "gemini": {
                "enabled": True,
                "versions": [v.id for v in PROVIDER_MODELS["gemini"].versions],
            },
        }
    
    sel = st.session_state["provider_selection"]
    
    if "active_run_id" not in st.session_state:
        st.session_state["active_run_id"] = ""

    # Sidebar content based on view
    if view == "Run Monitor":
        # Show configuration in sidebar for Run Monitor
        st.sidebar.header("Run Configuration")
        
        # DeepSeek
        with st.sidebar.expander("DeepSeek", expanded=False):
            sel["deepseek"]["enabled"] = st.checkbox(
                "Enable DeepSeek",
                value=sel["deepseek"]["enabled"],
                key="monitor_deepseek_enabled",
            )
        
        # GPT
        with st.sidebar.expander("ChatGPT (OpenAI)", expanded=False):
            sel["gpt"]["enabled"] = st.checkbox(
                "Enable GPT",
                value=sel["gpt"]["enabled"],
                key="monitor_gpt_enabled",
            )
            if sel["gpt"]["enabled"]:
                current = set(sel["gpt"]["versions"])
                versions = []
                for vcfg in PROVIDER_MODELS["gpt"].versions:
                    checked = st.checkbox(
                        f"{vcfg.label} ({vcfg.id})",
                        value=vcfg.id in current,
                        key=f"monitor_gpt_{vcfg.id}",
                    )
                    if checked:
                        versions.append(vcfg.id)
                sel["gpt"]["versions"] = versions
        
        # Claude
        with st.sidebar.expander("Claude (Anthropic)", expanded=False):
            sel["claude"]["enabled"] = st.checkbox(
                "Enable Claude",
                value=sel["claude"]["enabled"],
                key="monitor_claude_enabled",
            )
            if sel["claude"]["enabled"]:
                current = set(sel["claude"]["versions"])
                versions = []
                for vcfg in PROVIDER_MODELS["claude"].versions:
                    checked = st.checkbox(
                        f"{vcfg.label} ({vcfg.id})",
                        value=vcfg.id in current,
                        key=f"monitor_claude_{vcfg.id}",
                    )
                    if checked:
                        versions.append(vcfg.id)
                sel["claude"]["versions"] = versions
        
        # Gemini
        with st.sidebar.expander("Gemini (Google)", expanded=False):
            sel["gemini"]["enabled"] = st.checkbox(
                "Enable Gemini",
                value=sel["gemini"]["enabled"],
                key="monitor_gemini_enabled",
            )
            if sel["gemini"]["enabled"]:
                current = set(sel["gemini"]["versions"])
                versions = []
                for vcfg in PROVIDER_MODELS["gemini"].versions:
                    checked = st.checkbox(
                        f"{vcfg.label} ({vcfg.id})",
                        value=vcfg.id in current,
                        key=f"monitor_gemini_{vcfg.id}",
                    )
                    if checked:
                        versions.append(vcfg.id)
                sel["gemini"]["versions"] = versions
        
        st.session_state["provider_selection"] = sel
        
        # Run button in sidebar
        if st.sidebar.button("Run New Analysis", type="primary", use_container_width=True):
            run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            st.session_state["active_run_id"] = run_id
            thread = threading.Thread(
                target=start_background_run,
                args=(data_dir, prompts_file, keys_file, run_id, sel.copy()),
                daemon=True,
            )
            thread.start()
            st.sidebar.success(f"Started run: {run_id}")
            st.rerun()
        
        # Show run monitor in main area
        render_run_monitor(data_dir, prompts_file)
        return

    # Dashboard view: only show run history dropdown
    st.sidebar.header("Run History")
    run_ids = list_run_ids(data_dir)
    # Sort run_ids descending so the most recent is first
    sorted_run_ids = sorted(run_ids, reverse=True)
    display_map = {}
    display_options = []
    for rid in sorted_run_ids:
        if rid == "legacy":
            label = "Legacy baseline run"
        else:
            # Pretty label: YYYY-MM-DD HH:MM:SS (rid)
            try:
                dt = datetime.strptime(rid, "%Y%m%d-%H%M%S")
                label = dt.strftime("%Y-%m-%d %H:%M:%S") + f"  ({rid})"
            except Exception:
                label = rid
        display_map[label] = rid
        display_options.append(label)
    
    # Check if we should select a specific run (from redirect or URL param)
    default_index = 0
    selected_id = None
    if "run" in query_params:
        selected_id = query_params["run"]
    elif "selected_run_id" in st.session_state:
        selected_id = st.session_state["selected_run_id"]
        del st.session_state["selected_run_id"]
    
    if selected_id:
        for idx, label in enumerate(display_options):
            if display_map[label] == selected_id:
                default_index = idx
                break
    
    run_display = st.sidebar.selectbox(
        "Select Run",
        options=display_options,
        index=default_index if display_options else -1,
    )
    run_id = display_map.get(run_display, "")
    if not run_id:
        st.info("No runs found in metrics. Go to Run Monitor to start a new analysis.")
        return

    metrics_rows = read_jsonl(Path(data_dir) / "metrics" / f"{run_id}.jsonl")
    analysis_rows = read_jsonl(Path(data_dir) / "analysis" / f"{run_id}.jsonl")
    df = pd.DataFrame(metrics_rows)

    st.subheader("Filters")
    models = sorted(df["Model"].dropna().unique().tolist()) if "Model" in df.columns else []
    tones = sorted(df["PromptTone"].dropna().unique().tolist()) if "PromptTone" in df.columns else []
    versions = sorted(df["Version"].dropna().unique().tolist()) if "Version" in df.columns else []

    sel_models = st.multiselect("Models", models, default=models)
    sel_versions = st.multiselect("Versions", versions, default=versions)
    sel_tones = st.multiselect("Tones", tones, default=tones)

    if sel_models:
        df = df[df["Model"].isin(sel_models)]
    if sel_versions:
        df = df[df["Version"].isin(sel_versions)]
    if sel_tones:
        df = df[df["PromptTone"].isin(sel_tones)]

    tabs = st.tabs(["Overview", "Tone Impact", "Categories", "Safety", "Strategies", "Correlations", "Paired Tests", "Prompts"])
    with tabs[0]:
        st.subheader("Overview")
        cols = st.columns(4)
        def mean_or_nan(col):
            return float(df[col].mean()) if col in df.columns and not df.empty else float('nan')
        cols[0].metric("Sentiment (resp)", f"{mean_or_nan('Response_SentimentScore'):.3f}")
        cols[1].metric("Politeness (resp)", f"{mean_or_nan('Response_ValidatedPolitenessScore'):.3f}")
        cols[2].metric("Toxicity (resp)", f"{mean_or_nan('RoBERTa_Response_ToxicityScore'):.3f}")
        cols[3].metric("Verbosity (tokens)", f"{mean_or_nan('ResponseLength'):.1f}")
        st.subheader("Model × Tone comparisons")
        for metric in [
            "Response_SentimentScore",
            "Response_ValidatedPolitenessScore",
            "RoBERTa_Response_ToxicityScore",
            "ResponseLength",
        ]:
            if metric not in df.columns:
                continue
            gcols = [c for c in ["Model", "Version", "PromptTone"] if c in df.columns]
            g = df.groupby(gcols).agg({metric: "mean"}).reset_index()
            chart = alt.Chart(g).mark_bar().encode(
                x=alt.X("Model:N"),
                y=alt.Y(f"{metric}:Q"),
                color=alt.Color("PromptTone:N"),
                tooltip=["Model:N", "Version:N", "PromptTone:N", alt.Tooltip(f"{metric}:Q", format=".3f")]
            ).properties(width=600)
            st.altair_chart(chart, use_container_width=True)

    with tabs[1]:
        st.subheader("Tone deltas (polite − threatening)")
        if analysis_rows:
            art = analysis_rows[-1]
            deltas = pd.DataFrame(art.get("deltas", []))
            if not deltas.empty:
                deltas = deltas.reset_index(drop=True)
                deltas.index = deltas.index + 1
                st.dataframe(deltas)
                chart = alt.Chart(deltas).mark_bar().encode(
                    x=alt.X("Model:N"),
                    y=alt.Y("Delta_Sentiment_PoliteMinusThreatening:Q"),
                    color=alt.Color("Model:N")
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No delta artifacts found.")
        else:
            st.info("No analysis artifacts found.")

    with tabs[2]:
        st.subheader("Categories")
        if analysis_rows:
            art = analysis_rows[-1]
            cats = art.get("categories", {})
            for metric, obj in cats.items():
                st.write(metric)
                overall = pd.DataFrame(obj.get("overall", []))
                if not overall.empty:
                    cat_field = "TaskCategory" if "TaskCategory" in overall.columns else ("TaskDescription" if "TaskDescription" in overall.columns else None)
                    chart = alt.Chart(overall).mark_bar().encode(
                        x=alt.X(f"{cat_field}:N"), y=alt.Y(f"{metric}:Q"), color="PromptTone:N"
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)
                per_model = pd.DataFrame(obj.get("per_model", []))
                if not per_model.empty:
                    cat_field_pm = "TaskCategory" if "TaskCategory" in per_model.columns else ("TaskDescription" if "TaskDescription" in per_model.columns else None)
                    chart_pm = alt.Chart(per_model).mark_bar().encode(
                        x=alt.X(f"{cat_field_pm}:N"), y=alt.Y(f"{metric}:Q"), color="PromptTone:N", column="Model:N"
                    ).properties(height=300)
                    st.altair_chart(chart_pm, use_container_width=True)
        else:
            st.info("No analysis artifacts found.")

    with tabs[3]:
        st.subheader("Safety")
        if analysis_rows:
            art = analysis_rows[-1]
            rates = pd.DataFrame(art.get("safety", {}).get("rates", []))
            if not rates.empty:
                rates = rates.reset_index(drop=True)
                rates.index = rates.index + 1
                st.dataframe(rates)
                c1 = alt.Chart(rates).mark_bar().encode(
                    x="Model:N", y="Response_RefusalFlag:Q", color="PromptTone:N"
                ).properties(title="Refusal Rate")
                c2 = alt.Chart(rates).mark_bar().encode(
                    x="Model:N", y="Response_DisclaimerFlag:Q", color="PromptTone:N"
                ).properties(title="Disclaimer Rate")
                st.altair_chart(c1, use_container_width=True)
                st.altair_chart(c2, use_container_width=True)
        else:
            st.info("No analysis artifacts found.")

    with tabs[4]:
        st.subheader("Politeness Strategies")
        if analysis_rows:
            art = analysis_rows[-1]
            strat = pd.DataFrame(art.get("strategies", {}).get("summary", []))
            if not strat.empty:
                strat = strat.reset_index(drop=True)
                strat.index = strat.index + 1
                st.dataframe(strat)
                # Try common strategies if present
                keys = [k for k in strat.columns if k not in ["Model", "PromptTone"]]
                if keys:
                    long = strat.melt(id_vars=["Model", "PromptTone"], value_vars=keys, var_name="Strategy", value_name="Count")
                    chart = alt.Chart(long).mark_bar().encode(
                        x="Strategy:N", y="Count:Q", color="PromptTone:N", column="Model:N"
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No analysis artifacts found.")

    with tabs[5]:
        st.subheader("Correlations")
        if analysis_rows:
            art = analysis_rows[-1]
            corr = art.get("correlations", {})
            overall = corr.get("overall")
            if overall:
                df_corr = pd.DataFrame(overall)
                long = df_corr.reset_index().melt(id_vars="index", var_name="Variable", value_name="Correlation")
                heat = alt.Chart(long).mark_rect().encode(
                    x=alt.X("index:N", title=""), y=alt.Y("Variable:N", title=""), color=alt.Color("Correlation:Q", scale=alt.Scale(scheme='redblue', domain=[-1,1]))
                )
                text = alt.Chart(long).mark_text(baseline='middle').encode(
                    x="index:N", y="Variable:N", text=alt.Text("Correlation:Q", format=".2f"), color=alt.value("black")
                )
                st.altair_chart(heat + text, use_container_width=True)
            bt = corr.get("by_tone", {})
            if bt:
                tones = list(bt.keys())
                sel_tone = st.selectbox("Tone", options= tones)
                df_corr_t = pd.DataFrame(bt.get(sel_tone, {}))
                if not df_corr_t.empty:
                    long_t = df_corr_t.reset_index().melt(id_vars="index", var_name="Variable", value_name="Correlation")
                    heat_t = alt.Chart(long_t).mark_rect().encode(
                        x=alt.X("index:N", title=""), y=alt.Y("Variable:N", title=""), color=alt.Color("Correlation:Q", scale=alt.Scale(scheme='redblue', domain=[-1,1]))
                    )
                    text_t = alt.Chart(long_t).mark_text(baseline='middle').encode(
                        x="index:N", y="Variable:N", text=alt.Text("Correlation:Q", format=".2f"), color=alt.value("black")
                    )
                    st.altair_chart(heat_t + text_t, use_container_width=True)
        else:
            st.info("No analysis artifacts found.")

    with tabs[6]:
        st.subheader("Paired Tests")
        if analysis_rows:
            art = analysis_rows[-1]
            pt = art.get("paired_tests", {})
            metrics = list(pt.get("per_metric", {}).keys())
            if metrics:
                sel_metric = st.selectbox("Metric", options=metrics)
                rows = pd.DataFrame(pt.get("per_metric", {}).get(sel_metric, []))
                summary = pd.DataFrame(pt.get("summary", {}).get(sel_metric, []))
                if not rows.empty:
                    rows = rows.reset_index(drop=True)
                    rows.index = rows.index + 1
                    st.dataframe(rows)
                if not summary.empty:
                    summary = summary.reset_index(drop=True)
                    summary.index = summary.index + 1
                    st.dataframe(summary)
                    chart = alt.Chart(summary).mark_bar().encode(
                        x="Model:N", y="MeanDiff:Q", color="Model:N"
                    )
                    st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No analysis artifacts found.")

    with tabs[7]:
        st.subheader("Prompt Explorer")
        show_cols = [c for c in ["TaskID", "TaskDescription", "PromptTone", "Model", "Version", "ResponseLength", "Response_SentimentScore", "Response_ValidatedPolitenessScore", "RoBERTa_Response_ToxicityScore", "ResponseText"] if c in df.columns]
        df_show = df[show_cols] if show_cols else df
        df_show = df_show.reset_index(drop=True)
        df_show.index = df_show.index + 1
        st.dataframe(df_show)


if __name__ == "__main__":
    main()
