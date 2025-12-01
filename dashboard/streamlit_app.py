import os
import sys
import json
from pathlib import Path
import threading
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
    Dedicated page that shows live progress for the most recent / active run,
    broken down by model and by version (where selection metadata is available).
    """
    st.subheader("Run Monitor")
    active_run_id = st.session_state.get("active_run_id") or ""
    if not active_run_id:
        st.info("No active run detected. Start a new analysis from the Dashboard view.")
        return

    prog = read_run_progress(data_dir, active_run_id)
    if not prog:
        st.info(f"No progress metadata found for run_id={active_run_id}.")
        return

    cols = st.columns(4)
    cols[0].metric("Run ID", active_run_id)
    cols[1].metric("Status", prog.get("status", ""))
    cols[2].metric("Phase", prog.get("phase", ""))
    cols[3].metric("Total prompts", prog.get("total_prompts", 0))

    st.write(f"Last update (UTC): `{prog.get('updated_at', '')[:19]}`")

    # Compute per-model + per-version progress based on:
    # - recorded selection for this run
    # - current completions written for this run
    selection = prog.get("selection", {}) or {}
    per_model = prog.get("per_model", {}) or {}
    tasks = parse_tasks_file(prompts_file)
    num_tasks = len(tasks)

    if not selection:
        st.warning("No provider selection metadata stored for this run; showing per-model totals only.")
        # Fallback: show model-level progress only
        for model, stats in per_model.items():
            total = max(int(stats.get("total", 0)), 0)
            done = max(int(stats.get("completed", 0)) + int(stats.get("failed", 0)), 0)
            frac = float(done) / total if total > 0 else 0.0
            st.write(f"- `{model}` – {done}/{total} (all versions combined)")
            st.progress(min(frac, 1.0))
    else:
        st.markdown("### Per-model / per-version progress")
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

            st.markdown(f"#### Provider: `{provider_key}`")
            # Distribute model-level progress evenly across versions.
            # With your small test (2 prompts), this will show 0→1→2 as
            # the overall model finishes prompts.
            for ver in versions:
                expected = num_tasks
                actual = int(round(model_frac * expected))
                actual = min(actual, expected)
                frac = float(actual) / expected if expected > 0 else 0.0
                st.write(f"- `{provider_key}` / `{ver}` – {actual}/{expected}")
                st.progress(min(frac, 1.0))

    # Manual refresh to update progress without losing context.
    if prog.get("status") == "running":
        st.caption("Run in progress – click the button below to refresh progress.")
        if st.button("Refresh progress (monitor view)", key="refresh_progress_monitor"):
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
    view = st.sidebar.radio("View", ["Dashboard", "Run Monitor"], index=0)

    # ------------------------------------------------------------------
    # Sidebar: model / version selection (used for new runs)
    # Only show the detailed configuration when the user is in the
    # Dashboard view; keep it collapsed to make Run History easier to find.
    # ------------------------------------------------------------------
    st.sidebar.header("Run Configuration" if view == "Dashboard" else "Run Configuration (Dashboard only)")
    if "provider_selection" not in st.session_state:
        # Initialise with all configured versions enabled
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

    if view == "Dashboard":
        # DeepSeek (single default version today)
        with st.sidebar.expander("DeepSeek", expanded=False):
            sel["deepseek"]["enabled"] = st.checkbox(
                "Enable DeepSeek",
                value=sel["deepseek"]["enabled"],
                key="deepseek_enabled",
            )

        # GPT
        with st.sidebar.expander("ChatGPT (OpenAI)", expanded=False):
            sel["gpt"]["enabled"] = st.checkbox(
                "Enable GPT",
                value=sel["gpt"]["enabled"],
                key="gpt_enabled",
            )
            if sel["gpt"]["enabled"]:
                current = set(sel["gpt"]["versions"])
                versions = []
                for vcfg in PROVIDER_MODELS["gpt"].versions:
                    checked = st.checkbox(
                        f"{vcfg.label} ({vcfg.id})",
                        value=vcfg.id in current,
                        key=f"gpt_{vcfg.id}",
                    )
                    if checked:
                        versions.append(vcfg.id)
                sel["gpt"]["versions"] = versions

        # Claude
        with st.sidebar.expander("Claude (Anthropic)", expanded=False):
            sel["claude"]["enabled"] = st.checkbox(
                "Enable Claude",
                value=sel["claude"]["enabled"],
                key="claude_enabled",
            )
            if sel["claude"]["enabled"]:
                current = set(sel["claude"]["versions"])
                versions = []
                for vcfg in PROVIDER_MODELS["claude"].versions:
                    checked = st.checkbox(
                        f"{vcfg.label} ({vcfg.id})",
                        value=vcfg.id in current,
                        key=f"claude_{vcfg.id}",
                    )
                    if checked:
                        versions.append(vcfg.id)
                sel["claude"]["versions"] = versions

        # Gemini
        with st.sidebar.expander("Gemini (Google)", expanded=False):
            sel["gemini"]["enabled"] = st.checkbox(
                "Enable Gemini",
                value=sel["gemini"]["enabled"],
                key="gemini_enabled",
            )
            if sel["gemini"]["enabled"]:
                current = set(sel["gemini"]["versions"])
                versions = []
                for vcfg in PROVIDER_MODELS["gemini"].versions:
                    checked = st.checkbox(
                        f"{vcfg.label} ({vcfg.id})",
                        value=vcfg.id in current,
                        key=f"gemini_{vcfg.id}",
                    )
                    if checked:
                        versions.append(vcfg.id)
                sel["gemini"]["versions"] = versions

    st.session_state["provider_selection"] = sel

    # ------------------------------------------------------------------
    # Sidebar: run control (start new, retry missing)
    # ------------------------------------------------------------------
    if "active_run_id" not in st.session_state:
        st.session_state["active_run_id"] = ""

    if view == "Dashboard":
        if st.sidebar.button("Run New Analysis"):
            # Generate a run_id up front so both the background worker and UI
            # can agree on the same identifier.
            run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            st.session_state["active_run_id"] = run_id
            thread = threading.Thread(
                target=start_background_run,
                args=(data_dir, prompts_file, keys_file, run_id, sel.copy()),
                daemon=True,
            )
            thread.start()
            st.sidebar.success(f"Started new run: {run_id}")

    # Try to restore an active run if session state is empty by looking
    # at the most recent run from the logs directory.
    from glob import glob

    if not st.session_state["active_run_id"]:
        logs = sorted(glob(f"{data_dir}/logs/run_*.json"))
        if logs:
            # Pick the most recently modified log file.
            latest = max(logs, key=os.path.getmtime)
            rid = Path(latest).stem.replace("run_", "")
            st.session_state["active_run_id"] = rid

    # If user selected "Run Monitor", show a dedicated monitoring view
    # that focuses on live progress per model and per version.
    if view == "Run Monitor":
        render_run_monitor(data_dir, prompts_file)
        return

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
    run_display = st.sidebar.selectbox(
        "Run History",
        options=display_options,
        index=0 if display_options else -1,
    )
    run_id = display_map.get(run_display, "")
    if not run_id:
        st.info("No runs found in metrics. Ensure the pipeline has written to data/metrics/<run_id>.jsonl.")
        return

    # Sidebar: retry missing prompts for selected run (re-uses same run_id)
    if st.sidebar.button("Retry missing prompts for selected run"):
        retry_run_id = run_id
        st.session_state["active_run_id"] = retry_run_id
        thread = threading.Thread(
            target=start_background_run,
            args=(data_dir, prompts_file, keys_file, retry_run_id, sel.copy()),
            daemon=True,
        )
        thread.start()
        st.sidebar.success(f"Retrying missing prompts for run: {retry_run_id}")

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
