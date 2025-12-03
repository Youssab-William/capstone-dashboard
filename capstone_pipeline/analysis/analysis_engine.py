from typing import List, Dict, Any
import pandas as pd
from ..interfaces import AnalysisEngine
import logging


class ScriptAnalysisEngine(AnalysisEngine):
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def analyze(self, metrics_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics_rows:
            self.logger.info("analysis_no_rows")
            return {"figures": [], "highlights": []}
        df = pd.DataFrame(metrics_rows)
        self.logger.info(f"analysis_dataframe rows={len(df)} cols={list(df.columns)}")
        # Normalize category from TaskID if not provided
        if "TaskCategory" not in df.columns and "TaskID" in df.columns:
            def _parse_cat(x):
                s = "" if pd.isna(x) else str(x)
                import re
                m = re.match(r"^(.*)Task\d+$", s)
                if not m:
                    return ""
                prefix = m.group(1)
                return pd.Series(prefix).astype(str).str.replace(r"[^A-Za-z]", "", regex=True).iloc[0]
            df["TaskCategory"] = df["TaskID"].apply(_parse_cat)
        # Normalize tone labels
        if "PromptTone" in df.columns:
            df["PromptTone"] = df["PromptTone"].astype(str).str.strip().str.lower()
            df["PromptTone"] = df["PromptTone"].map({"polite": "Polite", "threatening": "Threatening"}).fillna(df["PromptTone"])
        summary = {}
        for metric in [
            "Response_SentimentScore",
            "Response_ValidatedPolitenessScore",
            "RoBERTa_Response_ToxicityScore",
            "ResponseLength",
        ]:
            if metric in df.columns:
                group_cols = [c for c in ["Model", "Version", "PromptTone"] if c in df.columns]
                toned = df[df["PromptTone"].isin(["Polite", "Threatening"]) ] if "PromptTone" in df.columns else df
                g = toned.groupby(group_cols).agg({metric: "mean"}).reset_index()
                summary[metric] = g.to_dict(orient="records")
        self.logger.info(f"analysis_summary_keys={list(summary.keys())}")
        deltas = []
        tone_col = "PromptTone"
        if {"Model", tone_col, "Response_SentimentScore"}.issubset(df.columns):
            index_cols = [c for c in ["Model", "Version"] if c in df.columns]
            pivot = df.pivot_table(index=index_cols, columns=tone_col, values="Response_SentimentScore", aggfunc="mean")
            pivot = pivot.reset_index()
            for _, row in pivot.iterrows():
                deltas.append({
                    "Model": row["Model"],
                    "Version": row.get("Version", ""),
                    "Delta_Sentiment_PoliteMinusThreatening": float((row.get("Polite") or 0) - (row.get("Threatening") or 0)),
                })
        self.logger.info(f"analysis_deltas_count={len(deltas)}")
        # Category-level means overall & per-model
        categories = {}
        # Determine category column
        category_col = "TaskCategory" if "TaskCategory" in df.columns else ("TaskDescription" if "TaskDescription" in df.columns else None)
        if category_col:
            toned = df[df[tone_col].isin(["Polite", "Threatening"]) ] if tone_col in df.columns else df
            for metric in ["Response_SentimentScore", "Response_ValidatedPolitenessScore", "RoBERTa_Response_ToxicityScore", "ResponseLength"]:
                if metric in df.columns:
                    overall = toned.groupby([category_col, tone_col]).agg({metric: "mean"}).reset_index()
                    per_model = toned.groupby([category_col, "Model", tone_col]).agg({metric: "mean"}).reset_index()
                    categories[metric] = {
                        "overall": overall.to_dict(orient="records"),
                        "per_model": per_model.to_dict(orient="records"),
                    }

        # Safety behavior rates
        safety = {}
        if {"Response_RefusalFlag", "Response_DisclaimerFlag"}.issubset(df.columns):
            safe_df = df[df[tone_col].isin(["Polite", "Threatening"]) ].copy() if tone_col in df.columns else df.copy()
            safe_df["Response_RefusalFlag"] = safe_df["Response_RefusalFlag"].astype(bool)
            safe_df["Response_DisclaimerFlag"] = safe_df["Response_DisclaimerFlag"].astype(bool)
            rates = safe_df.groupby(["Model", tone_col])[ ["Response_RefusalFlag", "Response_DisclaimerFlag"] ].mean().reset_index()
            safety["rates"] = rates.to_dict(orient="records")

        # Politeness strategies frequency parsing
        strategies = {}
        if "Response_ValidatedStrategies" in df.columns:
            toned = df[df[tone_col].isin(["Polite", "Threatening"]) ] if tone_col in df.columns else df
            def parse_strategies(s: str):
                out = {}
                if not isinstance(s, str) or not s.strip():
                    return out
                parts = [p.strip() for p in s.split(";") if p.strip()]
                for p in parts:
                    if ":" in p:
                        k, v = p.split(":", 1)
                        try:
                            out[k] = out.get(k, 0) + int(v)
                        except Exception:
                            pass
                return out
            parsed = toned["Response_ValidatedStrategies"].apply(parse_strategies)
            df_strat = pd.DataFrame(list(parsed)).fillna(0)
            df_strat["Model"] = toned["Model"]
            df_strat[tone_col] = toned[tone_col]
            strat_summary = df_strat.groupby(["Model", tone_col]).sum().reset_index()
            strategies["summary"] = strat_summary.to_dict(orient="records")

        # Correlations among sentiment, politeness, length (overall, by tone)
        correlations = {}
        corr_cols = [c for c in ["Response_SentimentScore", "Response_ValidatedPolitenessScore", "ResponseLength"] if c in df.columns]
        if len(corr_cols) >= 2:
            overall_corr = df[corr_cols].corr()
            correlations["overall"] = overall_corr.to_dict()
            by_tone = {}
            for t in df[tone_col].dropna().unique().tolist():
                sub = df[df[tone_col] == t]
                by_tone[t] = sub[corr_cols].corr().to_dict()
            correlations["by_tone"] = by_tone

        # Paired tests (Polite vs Threatening) per TaskID × Model
        paired_tests = {"per_metric": {}, "summary": {}}

        def compute_paired(df_in: pd.DataFrame, metric: str) -> Dict[str, any]:
            """
            Compute per-model paired differences between Polite and Threatening tones.

            Robust to cases where only one tone is present for a given model by
            simply skipping that model instead of raising a KeyError.
            """
            out_rows: List[Dict[str, Any]] = []
            summary_rows: List[Dict[str, Any]] = []
            for m in df_in["Model"].dropna().unique().tolist():
                sub = df_in[df_in["Model"] == m]
                piv = sub.pivot_table(
                    index=["TaskID"], columns=tone_col, values=metric, aggfunc="mean"
                )
                # Require BOTH Polite and Threatening columns to be present
                if not {"Polite", "Threatening"}.issubset(set(piv.columns)):
                    continue
                # Drop rows where either tone is missing
                piv = piv.dropna(subset=["Polite", "Threatening"], how="any")
                if piv.empty:
                    continue
                diffs = (piv["Polite"].fillna(0) - piv["Threatening"].fillna(0)).tolist()
                for tid, d in zip(piv.index.tolist(), diffs):
                    out_rows.append(
                        {
                            "Model": m,
                            "TaskID": tid,
                            "Metric": metric,
                            "Diff_PoliteMinusThreatening": float(d),
                        }
                    )
                mean_diff = float(pd.Series(diffs).mean()) if diffs else 0.0
                summary_rows.append(
                    {"Model": m, "Metric": metric, "MeanDiff": mean_diff, "N": len(diffs)}
                )
            return {"rows": out_rows, "summary": summary_rows}

        for metric in [
            "Response_SentimentScore",
            "Response_ValidatedPolitenessScore",
            "RoBERTa_Response_ToxicityScore",
            "ResponseLength",
        ]:
            if metric in df.columns:
                res = compute_paired(df, metric)
                paired_tests["per_metric"][metric] = res["rows"]
                paired_tests["summary"][metric] = res["summary"]

        highlights = []
        for d in deltas:
            if abs(d["Delta_Sentiment_PoliteMinusThreatening"]) > 0.2:
                highlights.append({"type": "sentiment_delta", **d})
        return {
            "summary": summary,
            "deltas": deltas,
            "categories": categories,
            "safety": safety,
            "strategies": strategies,
            "correlations": correlations,
            "paired_tests": paired_tests,
            "highlights": highlights,
        }
