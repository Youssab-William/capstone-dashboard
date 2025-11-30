import os
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import altair as alt
import os
import json
from capstone_pipeline.workflow.pipeline import Pipeline
from capstone_pipeline.storage.repository import JsonlStorage
from capstone_pipeline.metrics.metrics_engine import ScriptMetricsEngine
from capstone_pipeline.analysis.analysis_engine import ScriptAnalysisEngine
from capstone_pipeline.app import parse_tasks_file
from capstone_pipeline.providers.deepseek import DeepSeekProvider
from capstone_pipeline.providers.gpt import GPTProvider
from capstone_pipeline.providers.claude import ClaudeProvider
from capstone_pipeline.providers.gemini import GeminiProvider


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


def main():
    st.set_page_config(page_title="LLM Tone Dashboard", layout="wide")
    st.title("LLM Tone Effects Dashboard")

    data_dir = "data"
    prompts_file = "prompts.txt"
    keys_file = "data/keys.json"
    if st.sidebar.button("Run New Analysis"):
        try:
            if keys_file and os.path.exists(keys_file):
                with open(keys_file, "r", encoding="utf-8") as f:
                    keys = json.load(f)
                def set_key(env_name, value):
                    if value:
                        os.environ[env_name] = value
                set_key('DEEPSEEK_API_KEY', keys.get('deepseek_api_key') or keys.get('DEEPSEEK_API_KEY'))
                set_key('OPENAI_API_KEY', keys.get('openai_api_key') or keys.get('OPENAI_API_KEY'))
                set_key('ANTHROPIC_API_KEY', keys.get('anthropic_api_key') or keys.get('ANTHROPIC_API_KEY'))
                set_key('GEMINI_API_KEY', keys.get('gemini_api_key') or keys.get('GEMINI_API_KEY'))
                set_key('GOOGLE_API_KEY', keys.get('google_api_key') or keys.get('GOOGLE_API_KEY'))
            storage = JsonlStorage(data_dir)
            metrics = ScriptMetricsEngine()
            analysis = ScriptAnalysisEngine()
            pipeline = Pipeline(storage, metrics, analysis)
            tasks = parse_tasks_file(prompts_file)
            providers = [DeepSeekProvider(), GPTProvider(), ClaudeProvider(), GeminiProvider()]
            res = pipeline.run_all(tasks, providers)
            st.success(f"Pipeline completed: {res.get('run_id','')}")
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
    run_ids = list_run_ids(data_dir)
    display_map = {}
    for rid in run_ids:
        if rid == "legacy":
            display_map["20251126-080000"] = "legacy"
        else:
            display_map[rid] = rid
    display_options = list(display_map.keys())
    run_display = st.sidebar.selectbox("Run History", options=display_options, index=(0 if display_options else -1))
    run_id = display_map.get(run_display, run_display)
    if not run_id:
        st.info("No runs found in metrics. Ensure the pipeline has written to data/metrics/<run_id>.jsonl.")
        return

    metrics_rows = read_jsonl(Path(data_dir) / "metrics" / f"{run_id}.jsonl")
    analysis_rows = read_jsonl(Path(data_dir) / "analysis" / f"{run_id}.jsonl")
    df = pd.DataFrame(metrics_rows)

    st.subheader("Filters")
    models = sorted(df["Model"].dropna().unique().tolist()) if "Model" in df.columns else []
    tones = sorted(df["PromptTone"].dropna().unique().tolist()) if "PromptTone" in df.columns else []
    sel_models = st.multiselect("Models", models, default=models)
    sel_tones = st.multiselect("Tones", tones, default=tones)
    if sel_models:
        df = df[df["Model"].isin(sel_models)]
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
                    st.dataframe(rows)
                if not summary.empty:
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
        st.dataframe(df[show_cols] if show_cols else df)


if __name__ == "__main__":
    main()
