from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


@dataclass(frozen=True)
class VisualizationSpec:
    key: str
    label: str
    current_cols: tuple[str, ...]
    confidence_cols: tuple[str, ...]
    rank_col: str


VISUALIZATIONS: tuple[VisualizationSpec, ...] = (
    VisualizationSpec(
        key="polygon",
        label="Polygon",
        current_cols=("FR01", "FR19", "FR20"),
        confidence_cols=("FR03", "FR29", "FR30"),
        rank_col="FR18_01",
    ),
    VisualizationSpec(
        key="polygon_combined",
        label="Polygon Combined",
        current_cols=("FR08", "FR21", "FR22"),
        confidence_cols=("FR14", "FR31", "FR32"),
        rank_col="FR18_02",
    ),
    VisualizationSpec(
        key="polygon_binning",
        label="Polygon Binning",
        current_cols=("FR09", "FR23", "FR24"),
        confidence_cols=("FR15", "FR33", "FR34"),
        rank_col="FR18_03",
    ),
    VisualizationSpec(
        key="baseline",
        label="Baseline",
        current_cols=("FR06", "FR25", "FR26"),
        confidence_cols=("FR16", "FR35", "FR36"),
        rank_col="FR18_04",
    ),
    VisualizationSpec(
        key="baseline_map",
        label="Baseline + Map",
        current_cols=("FR07", "FR27", "FR28"),
        confidence_cols=("FR17", "FR37", "FR38"),
        rank_col="FR18_05",
    ),
)


CURRENT_LABELS = {1: "Yes", 2: "No", 3: "Unsure"}
CONF_LABELS = {
    1: "Not confident",
    2: "Slightly confident",
    3: "Moderately confident",
    4: "Very confident",
    5: "Fully confident",
}

FAMILIARITY_LABELS = {
    1: "Not familiar",
    2: "Slightly familiar",
    3: "Moderately familiar",
    4: "Very familiar",
    5: "Expert",
}


def read_tsv_auto(path: Path) -> pd.DataFrame:
    for encoding in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, sep="\t", encoding=encoding)
        except UnicodeError:
            continue
    return pd.read_csv(path, sep="\t")


def coalesce_numeric(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    subset = df.loc[:, columns].replace("", np.nan)
    for column in subset.columns:
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
    series = subset.bfill(axis=1).iloc[:, 0]
    series = series.replace(-9, np.nan)
    return series


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    ranked = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (idx, pval) in enumerate(ranked, start=1):
        adj = min(1.0, (m - rank + 1) * pval)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def cochran_q(binary: np.ndarray) -> tuple[float, float]:
    n, k = binary.shape
    col_sums = binary.sum(axis=0)
    row_sums = binary.sum(axis=1)
    total = col_sums.sum()
    numerator = (k - 1) * (k * np.sum(col_sums**2) - total**2)
    denominator = k * total - np.sum(row_sums**2)
    if denominator == 0:
        return np.nan, np.nan
    q = numerator / denominator
    p_val = stats.chi2.sf(q, k - 1)
    return float(q), float(p_val)


def mcnemar_exact(x: np.ndarray, y: np.ndarray) -> tuple[int, int, float]:
    b = int(np.sum((x == 0) & (y == 1)))
    c = int(np.sum((x == 1) & (y == 0)))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p_val = stats.binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue
    return b, c, float(p_val)


def mean_ci(series: pd.Series) -> tuple[float, float, float]:
    x = series.dropna().to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    delta = 1.96 * se
    return mean, mean - delta, mean + delta


def proportion_ci(series: pd.Series) -> tuple[float, float, float, int]:
    x = series.dropna().to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    p = float(np.mean(x))
    se = np.sqrt((p * (1 - p)) / n)
    delta = 1.96 * se
    return p, max(0.0, p - delta), min(1.0, p + delta), n


def build_dataset(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["CASE"] = pd.to_numeric(out["CASE"], errors="coerce")
    out["LASTPAGE"] = pd.to_numeric(out["LASTPAGE"], errors="coerce")
    out["FINISHED"] = pd.to_numeric(out["FINISHED"], errors="coerce")
    for spec in VISUALIZATIONS:
        out[f"{spec.key}_current"] = coalesce_numeric(out, spec.current_cols)
        out[f"{spec.key}_confidence"] = coalesce_numeric(out, spec.confidence_cols)
        out[f"{spec.key}_rank_pos"] = pd.to_numeric(out[spec.rank_col], errors="coerce")
    return out


def stacked_response_plot(df: pd.DataFrame, out_path: Path) -> None:
    labels = [spec.label for spec in VISUALIZATIONS]
    percentages = []
    for spec in VISUALIZATIONS:
        series = df[f"{spec.key}_current"].dropna()
        counts = series.value_counts().reindex([1, 2, 3], fill_value=0)
        pct = counts / counts.sum() * 100 if counts.sum() > 0 else counts
        percentages.append(pct.to_numpy())
    mat = np.array(percentages)
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(labels))
    colors = ["#4daf4a", "#377eb8", "#ff7f00"]
    for idx, code in enumerate([1, 2, 3]):
        ax.bar(labels, mat[:, idx], bottom=bottom, label=CURRENT_LABELS[code], color=colors[idx])
        bottom += mat[:, idx]
    ax.set_ylabel("Responses [%]")
    ax.set_title("Part 2 (Current Safety): Response Distribution by Visualization")
    ax.legend()
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def accuracy_plot(acc_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    y = acc_df["accuracy"] * 100
    yerr_low = (acc_df["accuracy"] - acc_df["ci_low"]) * 100
    yerr_high = (acc_df["ci_high"] - acc_df["accuracy"]) * 100
    ax.bar(acc_df["label"], y, yerr=[yerr_low, yerr_high], capsize=5, color="#377eb8")
    ax.set_ylabel("Correct answer rate (% No)")
    ax.set_title("Part 2 Accuracy by Visualization")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def confidence_plot(df: pd.DataFrame, out_path: Path) -> None:
    labels = [spec.label for spec in VISUALIZATIONS]
    data = [df[f"{spec.key}_confidence"].dropna().to_numpy() for spec in VISUALIZATIONS]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#a6cee3")
    ax.set_ylabel("Confidence (1=Not, 5=Fully)")
    ax.set_title("Part 2 Confidence by Visualization")
    ax.set_ylim(1, 5)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def ranking_plot(rank_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(rank_df["label"], rank_df["mean_rank"], color="#984ea3")
    ax.set_ylabel("Mean within-participant rank (1=easiest)")
    ax.set_title("Part 3 Ranking: Lower is Better")
    ax.set_ylim(1, 5)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def familiarity_plot(count_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(count_df["label"], count_df["percent"], color="#1b9e77")
    ax.set_ylabel("Participants [%]")
    ax.set_title("Part 1: Prior Familiarity with Flood/Geospatial Maps")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_analysis(data_path: Path, out_dir: Path) -> None:
    data = read_tsv_auto(data_path)
    df = build_dataset(data)
    out_dir.mkdir(parents=True, exist_ok=True)

    status_complete = df["STATUS"].astype(str).str.lower().eq("complete")
    completed = df[status_complete].copy()
    completed["familiarity"] = pd.to_numeric(completed.get("FR04"), errors="coerce").replace(-9, np.nan)

    current_cols = [f"{v.key}_current" for v in VISUALIZATIONS]
    conf_cols = [f"{v.key}_confidence" for v in VISUALIZATIONS]
    rank_cols = [f"{v.key}_rank_pos" for v in VISUALIZATIONS]

    part2_complete = completed.dropna(subset=current_cols).copy()
    part2_conf_complete = completed.dropna(subset=conf_cols).copy()
    part3_complete = completed.dropna(subset=rank_cols).copy()

    familiarity_counts = completed["familiarity"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    familiarity_total = int(familiarity_counts.sum())
    familiarity_df = pd.DataFrame(
        {
            "code": familiarity_counts.index.astype(int),
            "label": [FAMILIARITY_LABELS[int(i)] for i in familiarity_counts.index],
            "count": familiarity_counts.values.astype(int),
            "percent": (familiarity_counts.values / familiarity_total * 100) if familiarity_total else np.nan,
        }
    )

    for spec in VISUALIZATIONS:
        part2_complete[f"{spec.key}_correct"] = (part2_complete[f"{spec.key}_current"] == 2).astype(int)

    accuracy_rows = []
    for spec in VISUALIZATIONS:
        p, lo, hi, n = proportion_ci(part2_complete[f"{spec.key}_correct"])
        accuracy_rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "accuracy": p,
                "ci_low": lo,
                "ci_high": hi,
                "n": n,
            }
        )
    accuracy_df = pd.DataFrame(accuracy_rows)

    binary_matrix = np.column_stack([part2_complete[f"{v.key}_correct"].to_numpy() for v in VISUALIZATIONS])
    q_stat, q_p = cochran_q(binary_matrix)

    mcnemar_rows = []
    raw_p = []
    pairs = list(combinations(VISUALIZATIONS, 2))
    for a, b in pairs:
        x = part2_complete[f"{a.key}_correct"].to_numpy()
        y = part2_complete[f"{b.key}_correct"].to_numpy()
        b01, c10, p_val = mcnemar_exact(x, y)
        raw_p.append(p_val)
        mcnemar_rows.append(
            {
                "comparison": f"{a.label} vs {b.label}",
                "b_incorrect_to_correct": b01,
                "c_correct_to_incorrect": c10,
                "p_raw": p_val,
            }
        )
    mcnemar_df = pd.DataFrame(mcnemar_rows)
    if not mcnemar_df.empty:
        mcnemar_df["p_holm"] = holm_adjust(raw_p)

    conf_matrix = np.column_stack([part2_conf_complete[f"{v.key}_confidence"].to_numpy() for v in VISUALIZATIONS])
    conf_friedman = stats.friedmanchisquare(*[conf_matrix[:, i] for i in range(conf_matrix.shape[1])])

    conf_pair_rows = []
    conf_p = []
    for a, b in pairs:
        x = part2_conf_complete[f"{a.key}_confidence"].to_numpy()
        y = part2_conf_complete[f"{b.key}_confidence"].to_numpy()
        try:
            p_val = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue
        except ValueError:
            p_val = 1.0
        conf_p.append(float(p_val))
        conf_pair_rows.append({"comparison": f"{a.label} vs {b.label}", "p_raw": float(p_val)})
    conf_pairs_df = pd.DataFrame(conf_pair_rows)
    if not conf_pairs_df.empty:
        conf_pairs_df["p_holm"] = holm_adjust(conf_p)

    rank_positions = part3_complete[rank_cols].copy()
    rank_within = rank_positions.rank(axis=1, method="average", ascending=True)
    rank_within.columns = [f"{v.key}_rank" for v in VISUALIZATIONS]
    rank_matrix = rank_within.to_numpy()
    rank_friedman = stats.friedmanchisquare(*[rank_matrix[:, i] for i in range(rank_matrix.shape[1])])

    rank_pair_rows = []
    rank_p = []
    for a, b in pairs:
        x = rank_within[f"{a.key}_rank"].to_numpy()
        y = rank_within[f"{b.key}_rank"].to_numpy()
        try:
            p_val = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue
        except ValueError:
            p_val = 1.0
        rank_p.append(float(p_val))
        rank_pair_rows.append({"comparison": f"{a.label} vs {b.label}", "p_raw": float(p_val)})
    rank_pairs_df = pd.DataFrame(rank_pair_rows)
    if not rank_pairs_df.empty:
        rank_pairs_df["p_holm"] = holm_adjust(rank_p)

    rank_summary = []
    for spec in VISUALIZATIONS:
        series = rank_within[f"{spec.key}_rank"]
        mean, lo, hi = mean_ci(series)
        rank_summary.append({"key": spec.key, "label": spec.label, "mean_rank": mean, "ci_low": lo, "ci_high": hi})
    rank_df = pd.DataFrame(rank_summary).sort_values("mean_rank")

    response_dist_rows = []
    for spec in VISUALIZATIONS:
        counts = part2_complete[f"{spec.key}_current"].value_counts().reindex([1, 2, 3], fill_value=0)
        total = int(counts.sum())
        for code, count in counts.items():
            response_dist_rows.append(
                {
                    "visualization": spec.label,
                    "response_code": int(code),
                    "response_label": CURRENT_LABELS[int(code)],
                    "count": int(count),
                    "percent": (count / total * 100) if total else np.nan,
                }
            )
    response_dist_df = pd.DataFrame(response_dist_rows)

    confidence_summary_rows = []
    for spec in VISUALIZATIONS:
        series = part2_conf_complete[f"{spec.key}_confidence"]
        mean, lo, hi = mean_ci(series)
        confidence_summary_rows.append({"visualization": spec.label, "mean": mean, "ci_low": lo, "ci_high": hi})
    confidence_summary_df = pd.DataFrame(confidence_summary_rows).sort_values("mean", ascending=False)

    familiarity_plot(familiarity_df, out_dir / "part1_familiarity_distribution.png")
    stacked_response_plot(part2_complete, out_dir / "part2_response_distribution.png")
    accuracy_plot(accuracy_df, out_dir / "part2_accuracy.png")
    confidence_plot(part2_conf_complete, out_dir / "part2_confidence_boxplot.png")
    ranking_plot(rank_df, out_dir / "part3_mean_rank.png")

    familiarity_df.to_csv(out_dir / "part1_familiarity_table.csv", index=False)
    accuracy_df.to_csv(out_dir / "part2_accuracy_table.csv", index=False)
    response_dist_df.to_csv(out_dir / "part2_response_distribution_table.csv", index=False)
    mcnemar_df.to_csv(out_dir / "part2_mcnemar_pairwise.csv", index=False)
    confidence_summary_df.to_csv(out_dir / "part2_confidence_summary.csv", index=False)
    conf_pairs_df.to_csv(out_dir / "part2_confidence_pairwise_wilcoxon.csv", index=False)
    rank_df.to_csv(out_dir / "part3_rank_summary.csv", index=False)
    rank_pairs_df.to_csv(out_dir / "part3_rank_pairwise_wilcoxon.csv", index=False)

    assumptions = [
        "Current safety coding assumed as 1=Yes, 2=No, 3=Unsure.",
        "Correct answer assumed to be No (code 2), based on survey design in the report.",
        "Confidence coding assumed as 1=Not confident to 5=Fully confident.",
        "Part 3 ordering interpreted as lower position value = easier (SoSci left-to-right scale).",
        "Primary inferential analyses restricted to STATUS='complete'.",
    ]

    summary_lines = []
    summary_lines.append("# Survey Analysis Summary")
    summary_lines.append("")
    summary_lines.append("## Sample")
    summary_lines.append(f"- Total rows in dataset: {len(df)}")
    summary_lines.append(f"- Completed interviews (STATUS='complete'): {len(completed)}")
    summary_lines.append(f"- Part 1 familiarity responses: {familiarity_total}")
    summary_lines.append(f"- Part 2 complete cases (all 5 current answers): {len(part2_complete)}")
    summary_lines.append(f"- Part 2 confidence complete cases: {len(part2_conf_complete)}")
    summary_lines.append(f"- Part 3 ranking complete cases: {len(part3_complete)}")
    summary_lines.append("")
    summary_lines.append("## Part 1: Prior Familiarity")
    for _, row in familiarity_df.iterrows():
        summary_lines.append(f"- {row['label']}: {int(row['count'])} ({row['percent']:.1f}%)")
    summary_lines.append("")
    summary_lines.append("## Part 2: Current Safety Accuracy")
    for _, row in accuracy_df.sort_values("accuracy", ascending=False).iterrows():
        summary_lines.append(
            f"- {row['label']}: {row['accuracy']*100:.1f}% correct (95% CI {row['ci_low']*100:.1f}–{row['ci_high']*100:.1f}, n={int(row['n'])})"
        )
    summary_lines.append(f"- Cochran's Q across visualizations: Q={q_stat:.3f}, p={q_p:.4g}")
    summary_lines.append("")
    summary_lines.append("### Pairwise McNemar (Holm-adjusted)")
    for _, row in mcnemar_df.sort_values("p_holm").iterrows():
        summary_lines.append(
            f"- {row['comparison']}: p_raw={row['p_raw']:.4g}, p_holm={row['p_holm']:.4g}, b={int(row['b_incorrect_to_correct'])}, c={int(row['c_correct_to_incorrect'])}"
        )
    summary_lines.append("")
    summary_lines.append("## Part 2: Confidence")
    for _, row in confidence_summary_df.iterrows():
        summary_lines.append(
            f"- {row['visualization']}: mean={row['mean']:.2f} (95% CI {row['ci_low']:.2f}–{row['ci_high']:.2f})"
        )
    summary_lines.append(
        f"- Friedman test across visualizations: chi2={conf_friedman.statistic:.3f}, p={conf_friedman.pvalue:.4g}"
    )
    summary_lines.append("")
    summary_lines.append("### Pairwise Wilcoxon for Confidence (Holm-adjusted)")
    for _, row in conf_pairs_df.sort_values("p_holm").iterrows():
        summary_lines.append(f"- {row['comparison']}: p_raw={row['p_raw']:.4g}, p_holm={row['p_holm']:.4g}")
    summary_lines.append("")
    summary_lines.append("## Part 3: Final Ranking")
    for _, row in rank_df.iterrows():
        summary_lines.append(
            f"- {row['label']}: mean rank={row['mean_rank']:.2f} (95% CI {row['ci_low']:.2f}–{row['ci_high']:.2f})"
        )
    summary_lines.append(
        f"- Friedman test across visualization ranks: chi2={rank_friedman.statistic:.3f}, p={rank_friedman.pvalue:.4g}"
    )
    summary_lines.append("")
    summary_lines.append("### Pairwise Wilcoxon for Ranking (Holm-adjusted)")
    for _, row in rank_pairs_df.sort_values("p_holm").iterrows():
        summary_lines.append(f"- {row['comparison']}: p_raw={row['p_raw']:.4g}, p_holm={row['p_holm']:.4g}")
    summary_lines.append("")
    summary_lines.append("## Assumptions")
    for item in assumptions:
        summary_lines.append(f"- {item}")

    (out_dir / "survey_analysis_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Analyze flood visualization survey results.")
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir / "data_floodvis_2025-12-10_22-59.csv",
        help="Path to SoSci data export TSV file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=script_dir / "survey_analysis",
        help="Directory where analysis outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(args.data, args.outdir)
    print(f"Analysis complete. Outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()