import os
import re
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_category(task_id: str) -> str:
    s = "" if pd.isna(task_id) else str(task_id)
    m = re.match(r"^(.*)Task\d+$", s)
    if not m:
        return ""
    prefix = m.group(1)
    return re.sub(r"[^A-Za-z]", "", prefix)


def normalize_tone(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if s.lower().startswith("polite"):
        return "Polite"
    if s.lower().startswith("threat"):
        return "Threatening"
    return s


def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "TaskCategory" not in df.columns:
        df["TaskCategory"] = df["TaskID"].apply(parse_category) if "TaskID" in df.columns else ""
    if "PromptTone" in df.columns:
        df["PromptTone"] = df["PromptTone"].apply(normalize_tone)
    return df


def fig_overall_means(df: pd.DataFrame, out_dir: Path, metric: str):
    if metric not in df.columns:
        return
    plt.figure(figsize=(12, 6))
    g = df.groupby(["TaskCategory", "PromptTone"]).agg({metric: "mean"}).reset_index()
    sns.barplot(data=g, x="TaskCategory", y=metric, hue="PromptTone")
    plt.title(f"Overall means by TaskCategory × Tone — {metric}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"overall_means_{metric}.png")
    plt.close()


def fig_per_model_means(df: pd.DataFrame, out_dir: Path, metric: str):
    if metric not in df.columns:
        return
    g = df.groupby(["TaskCategory", "Model", "PromptTone"]).agg({metric: "mean"}).reset_index()
    models = sorted(g["Model"].dropna().unique().tolist())
    for m in models:
        plt.figure(figsize=(12, 6))
        sub = g[g["Model"] == m]
        sns.barplot(data=sub, x="TaskCategory", y=metric, hue="PromptTone")
        plt.title(f"Per-model means by TaskCategory × Tone — {metric} — {m}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"per_model_means_{metric}_{m}.png")
        plt.close()


def fig_deltas_by_category(df: pd.DataFrame, out_dir: Path, metric: str):
    if metric not in df.columns:
        return
    pvt = df.pivot_table(index=["TaskCategory"], columns="PromptTone", values=metric, aggfunc="mean")
    pvt = pvt.reset_index()
    polite = pvt["Polite"] if "Polite" in pvt.columns else pd.Series(0, index=pvt.index)
    threat = pvt["Threatening"] if "Threatening" in pvt.columns else pd.Series(0, index=pvt.index)
    pvt["Delta_PoliteMinusThreatening"] = polite.fillna(0) - threat.fillna(0)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=pvt, x="TaskCategory", y="Delta_PoliteMinusThreatening")
    plt.axhline(0, color="gray", linewidth=1)
    plt.title(f"Delta (Polite − Threatening) by TaskCategory — {metric}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"delta_by_category_{metric}.png")
    plt.close()


def fig_correlations(df: pd.DataFrame, out_dir: Path, cols: list, name: str):
    sub = df[cols].dropna()
    if sub.empty:
        return
    corr = sub.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.title(f"Correlation Heatmap — {name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"correlations_{name}.png")
    plt.close()


def fig_safety(df: pd.DataFrame, out_dir: Path):
    if "Response_RefusalFlag" not in df.columns or "Response_DisclaimerFlag" not in df.columns:
        return
    safe = df.copy()
    safe["Response_RefusalFlag"] = safe["Response_RefusalFlag"].astype(bool)
    safe["Response_DisclaimerFlag"] = safe["Response_DisclaimerFlag"].astype(bool)
    rates = safe.groupby(["Model", "PromptTone"]).agg({
        "Response_RefusalFlag": "mean",
        "Response_DisclaimerFlag": "mean",
    }).reset_index()
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    sns.barplot(data=rates, x="Model", y="Response_RefusalFlag", hue="PromptTone", ax=ax1)
    ax1.set_title("Refusal Rate")
    plt.xticks(rotation=45, ha="right")
    ax2 = plt.subplot(1, 2, 2)
    sns.barplot(data=rates, x="Model", y="Response_DisclaimerFlag", hue="PromptTone", ax=ax2)
    ax2.set_title("Disclaimer Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "safety_rates.png")
    plt.close()


def fig_strategies(df: pd.DataFrame, out_dir: Path):
    col = "Response_ValidatedStrategies"
    if col not in df.columns:
        return
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
    parsed = df[col].apply(parse_strategies)
    strat_df = pd.DataFrame(list(parsed)).fillna(0)
    if strat_df.empty:
        return
    strat_df["Model"] = df["Model"]
    strat_df["PromptTone"] = df["PromptTone"]
    long = strat_df.melt(id_vars=["Model", "PromptTone"], var_name="Strategy", value_name="Count")
    plt.figure(figsize=(14, 6))
    sns.barplot(data=long, x="Strategy", y="Count", hue="PromptTone")
    plt.xticks(rotation=45, ha="right")
    plt.title("Politeness Strategies Frequency by Tone (aggregated)")
    plt.tight_layout()
    plt.savefig(out_dir / "politeness_strategies.png")
    plt.close()


def main():
    csv_path = Path("legacy/final_dataset.csv")
    out_dir = Path("legacy/phase_3_legacy_code/Statistical Analysis Scripts/analysis_outputs_imgs")
    ensure_dir(out_dir)
    df = load_df(csv_path)

    core_metrics = [
        "Response_SentimentScore",
        "Response_ValidatedPolitenessScore",
        "RoBERTa_Response_ToxicityScore",
        "ResponseLength",
    ]
    for m in core_metrics:
        fig_overall_means(df, out_dir, m)
        fig_per_model_means(df, out_dir, m)
        fig_deltas_by_category(df, out_dir, m)

    fig_correlations(df, out_dir, [
        "Response_SentimentScore",
        "Response_ValidatedPolitenessScore",
        "ResponseLength",
    ], name="response")

    if {"Prompt_SentimentScore"}.issubset(df.columns):
        fig_correlations(df, out_dir, [
            "Prompt_SentimentScore",
            "Response_SentimentScore",
            "ResponseLength",
        ], name="prompt_response")

    fig_safety(df, out_dir)
    fig_strategies(df, out_dir)

    print("Figures generated in", out_dir)


if __name__ == "__main__":
    main()
