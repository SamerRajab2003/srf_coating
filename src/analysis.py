"""Statistical reduction, hypothesis testing, and single-cavity reporting.

Reads raw Monte Carlo simulation outputs for an individual cavity run and computes:
  1. Summary statistics for quality factor, frequency detuning, and quench factors.
  2. Loaded quality factor and power reflection from coupler mismatch.
  3. Parametric and non-parametric hypothesis tests (Welch ANOVA, Levene, Games-Howell).
  4. Diagnostic distribution figures (boxplots, normal Q-Q plots, parameter error bars).
  5. Invariant field metrics (effective loss area, peakiness) from raw mesh fields.
"""

from itertools import combinations
import logging
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import probplot, studentized_range

from src.errors import PhysicsError
from src.physics import compute_field_metrics_from_raw

logger = logging.getLogger(__name__)


def analyze_run(run_dir: Path) -> None:
    """Performs statistical reduction and diagnostic plotting for a single run.

    Args:
        run_dir: Path to the target run directory containing 'all_samples_raw.csv'
            and optional 'extracted_fields.npz'.

    Raises:
        FileNotFoundError: If 'all_samples_raw.csv' does not exist in run_dir.
        PhysicsError: If numerical arrays contain non-physical or degenerate data.
    """
    run_dir = Path(run_dir)
    csv_path = run_dir / "all_samples_raw.csv"
    if not csv_path.exists():
        logger.error("Raw samples CSV not found at %s", csv_path)
        return

    logger.info("Loading raw samples from %s", csv_path)
    df = pd.read_csv(csv_path)
    length_scales_mm: List[float] = sorted(
        df["length_scale_mm"].unique().tolist())
    q_uniform: float = float(df["Q"].iloc[0] / df["Q_normalized"].iloc[0])

    results_db: Dict[float, np.ndarray] = {
        l_mm: df[df["length_scale_mm"] == l_mm]["Q_normalized"].to_numpy()
        for l_mm in length_scales_mm
    }
    output_dir = str(run_dir)

    # =========================================================================
    # 1. Summary Statistics (Q, Frequency Shift, Hot Spot, Loaded Parameters)
    # =========================================================================
    summary_rows = []
    for l_mm in length_scales_mm:
        vals_q = results_db[l_mm]
        sub = df[df["length_scale_mm"] == l_mm]
        n_samples = len(vals_q)
        mean_q = float(np.mean(vals_q))
        std_q = float(np.std(vals_q, ddof=1))
        t_crit = float(stats.t.ppf(0.975, n_samples - 1))

        # Critical values for 95% Chi-squared confidence interval on standard deviation
        chi2_low = float(stats.chi2.ppf(0.975, n_samples - 1))
        chi2_high = float(stats.chi2.ppf(0.025, n_samples - 1))
        std_ci_low = float(std_q * np.sqrt((n_samples - 1) / chi2_low))
        std_ci_high = float(std_q * np.sqrt((n_samples - 1) / chi2_high))

        row = {
            "length_scale_mm": l_mm,
            "samples": n_samples,
            "mean_Q_normalized": mean_q,
            "mean_ci95_low": mean_q - t_crit * std_q / np.sqrt(n_samples),
            "mean_ci95_high": mean_q + t_crit * std_q / np.sqrt(n_samples),
            "std_Q_normalized": std_q,
            "std_ci95_low": std_ci_low,
            "std_ci95_high": std_ci_high,
            "minimum_Q_normalized": float(np.min(vals_q)),
            "maximum_Q_normalized": float(np.max(vals_q)),
            "minimum_deviation_pct": float(100.0 * (np.min(vals_q) - 1.0)),
            "maximum_deviation_pct": float(100.0 * (np.max(vals_q) - 1.0)),
        }

        # Task 11: Frequency detuning observables
        if "delta_f_hz" in sub.columns:
            row["mean_delta_f_hz"] = float(sub["delta_f_hz"].mean())
            row["std_delta_f_hz"] = float(sub["delta_f_hz"].std(ddof=1))

        # Task 11: Localized dissipation ratio (quench threat)
        if "f_hot" in sub.columns:
            row["mean_f_hot"] = float(sub["f_hot"].mean())
            row["max_f_hot"] = float(sub["f_hot"].max())
            row["q95_f_hot"] = float(sub["f_hot"].quantile(0.95))

        # Task 11: Loaded Q and power reflection from coupler mismatch
        # Assumes coupler external Q is matched to nominal coating (Q_ext = Q_uniform)
        q_norm_arr = vals_q
        q_loaded_arr = q_uniform * (q_norm_arr / (1.0 + q_norm_arr))
        reflected_power_pct = ((1.0 - q_norm_arr) /
                               (1.0 + q_norm_arr)) ** 2 * 100.0

        row["mean_Q_loaded"] = float(np.mean(q_loaded_arr))
        row["std_Q_loaded"] = float(np.std(q_loaded_arr, ddof=1))
        row["max_power_reflected_pct"] = float(np.max(reflected_power_pct))

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(Path(output_dir) / "statistics_summary.csv", index=False)
    logger.info("\n========== STATISTICAL SUMMARY ==========\n%s",
                summary_df.to_string(index=False))

    # =========================================================================
    # 2. Overall Levene Test (Homogeneity of Variance across Length Scales)
    # =========================================================================
    groups = [results_db[p] for p in length_scales_mm]
    levene_stat, levene_p = stats.levene(*groups, center="median")
    logger.info(
        "\n========== OVERALL LEVENE TEST ==========\nOverall Levene p-value: %.6e", levene_p)

    # =========================================================================
    # 3. Pairwise Levene Tests with Holm-Bonferroni Family-Wise Correction
    # =========================================================================

    pairwise_levene = []
    for l_a, l_b in combinations(length_scales_mm, 2):
        stat, p_val = stats.levene(
            results_db[l_a], results_db[l_b], center="median")
        pairwise_levene.append({
            "group_a_mm": l_a,
            "group_b_mm": l_b,
            "statistic": float(stat),
            "p_value": float(p_val)
        })
    pair_df = pd.DataFrame(pairwise_levene)
    sort_order = np.argsort(pair_df["p_value"].to_numpy())
    sorted_p = pair_df.loc[sort_order, "p_value"].to_numpy()
    adjusted_sorted = np.empty(len(sorted_p))
    running_max = 0.0
    for idx, p_val in enumerate(sorted_p):
        adjusted = min(1.0, (len(sorted_p) - idx) * p_val)
        running_max = max(running_max, adjusted)
        adjusted_sorted[idx] = running_max
    adjusted_p = np.empty(len(sorted_p))
    adjusted_p[sort_order] = adjusted_sorted
    pair_df["holm_adjusted_p"] = adjusted_p
    pair_df["reject_at_0.05"] = pair_df["holm_adjusted_p"] < 0.05
    pair_df.to_csv(Path(output_dir) / "levene_pairwise.csv", index=False)
    logger.info("\n========== PAIRWISE LEVENE TESTS ==========\n%s",
                pair_df.to_string(index=False))

    # =========================================================================
    # 4. Welch ANOVA (Equality of Means under Heteroscedasticity)
    # =========================================================================
    n_arr = np.array([len(g) for g in groups], dtype=float)
    means = np.array([np.mean(g) for g in groups])
    variances = np.array([np.var(g, ddof=1) for g in groups])
    weights = n_arr / variances
    weight_sum = np.sum(weights)
    weighted_mean = np.sum(weights * means) / weight_sum
    k_groups = len(groups)
    numerator = np.sum(weights * (means - weighted_mean)
                       ** 2) / (k_groups - 1.0)
    corr_sum = np.sum((1.0 - weights / weight_sum) ** 2 / (n_arr - 1.0))
    corr_term = 1.0 + 2.0 * (k_groups - 2.0) * corr_sum / (k_groups ** 2 - 1.0)
    f_stat = numerator / corr_term
    deg_freedom_1 = k_groups - 1.0
    deg_freedom_2 = (k_groups ** 2 - 1.0) / (3.0 * corr_sum)
    welch_p = float(stats.f.sf(f_stat, deg_freedom_1, deg_freedom_2))

    overall_df = pd.DataFrame([
        {"test": "Overall Levene", "statistic": float(levene_stat), "p_value": float(levene_p),
         "reject_at_0.05": bool(levene_p < 0.05)},
        {"test": "Welch ANOVA", "statistic": float(f_stat), "p_value": welch_p,
         "reject_at_0.05": bool(welch_p < 0.05)},
    ])
    overall_df.to_csv(Path(output_dir) /
                      "hypothesis_tests_overall.csv", index=False)
    logger.info(
        "\n========== WELCH ANOVA ==========\nWelch ANOVA p-value: %.6e", welch_p)

    # =========================================================================
    # 5. Games-Howell Post-Hoc Pairwise Comparisons
    # =========================================================================
    gh_rows = []
    for i, j in combinations(range(k_groups), 2):
        mean_diff = means[i] - means[j]
        var_term = variances[i] / n_arr[i] + variances[j] / n_arr[j]
        q_stat = abs(mean_diff) / np.sqrt(0.5 * var_term)
        deg_free_num = var_term ** 2
        deg_free_den = (
            (variances[i] / n_arr[i]) ** 2 / (n_arr[i] - 1.0) +
            (variances[j] / n_arr[j]) ** 2 / (n_arr[j] - 1.0)
        )
        deg_freedom = deg_free_num / deg_free_den

        p_value_gh = float(studentized_range.sf(q_stat, k_groups, deg_freedom))
        gh_rows.append({
            "group_a_mm": length_scales_mm[i],
            "group_b_mm": length_scales_mm[j],
            "mean_difference": float(mean_diff),
            "q_statistic": float(q_stat),
            "degrees_freedom": float(deg_freedom),
            "p_value": p_value_gh,
            "reject_at_0.05": bool(p_value_gh < 0.05)
        })
    gh_df = pd.DataFrame(gh_rows)
    gh_df.to_csv(Path(output_dir) / "games_howell.csv", index=False)
    logger.info("\n========== GAMES-HOWELL TEST ==========\n%s",
                gh_df.to_string(index=False))

    # =========================================================================
    # 6. Diagnostic Visualizations (Boxplots, Q-Q Plots, Parameter Error Bars)
    # =========================================================================
    logger.info("Generating plots...")

    # 6A. Boxplots
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        groups,
        positions=length_scales_mm,
        widths=1.5,
        patch_artist=True,
        boxprops=dict(facecolor="wheat", color="blue"),
        medianprops=dict(color="orange", linewidth=2),
        flierprops=dict(marker="o", color="purple", markersize=4, alpha=0.5)
    )
    plt.axhline(1.0, color="gray", linestyle="--",
                linewidth=1, label="Uniform Coating")
    plt.title("Normalized Quality Factor Distribution vs Length Scale",
              fontsize=14, fontweight="bold")
    plt.xlabel("Length Scale / Cavity Length [%]", fontsize=12)
    plt.ylabel(r"$Q / Q_u$", fontsize=12)
    plt.xticks(length_scales_mm, length_scales_mm)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/boxplots.png", dpi=300)
    plt.close()

# 6B. Normal Q-Q Plots
    n_plots = len(length_scales_mm)
    ncols = min(3, n_plots)
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(
        5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for idx, l_mm in enumerate(length_scales_mm):
        probplot(results_db[l_mm], dist="norm", plot=axes_flat[idx])
        axes_flat[idx].set_title(f"{l_mm} mm")
        axes_flat[idx].grid(True, linestyle="--", alpha=0.3)

    # Hide any unused subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Normal Q-Q Plots of Normalized Quality Factors", fontsize=14)
    fig.tight_layout()
    plt.savefig(f"{output_dir}/qq_plots.png", dpi=300)
    plt.close()

    # 6C. Distribution Parameter Error Bars
    means_arr = summary_df["mean_Q_normalized"].to_numpy()
    stds_arr = summary_df["std_Q_normalized"].to_numpy()
    mean_errors = np.vstack((
        means_arr - summary_df["mean_ci95_low"].to_numpy(),
        summary_df["mean_ci95_high"].to_numpy() - means_arr
    ))
    std_errors = np.vstack((
        stds_arr - summary_df["std_ci95_low"].to_numpy(),
        summary_df["std_ci95_high"].to_numpy() - stds_arr
    ))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].errorbar(length_scales_mm, means_arr,
                     yerr=mean_errors, fmt="o", capsize=5)
    axes[0].axhline(1.0, color="gray", linestyle="--", label="Uniform coating")
    axes[0].set_xlabel("Length Scale / Cavity Length [%]")
    axes[0].set_ylabel(r"$\mu_Q / Q_u$")
    axes[0].set_xticks(length_scales_mm)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].errorbar(length_scales_mm, stds_arr,
                     yerr=std_errors, fmt="o", capsize=5)
    axes[1].set_xlabel("Length Scale / Cavity Length [%]")
    axes[1].set_ylabel(r"$\sigma_Q / Q_u$")
    axes[1].set_xticks(length_scales_mm)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Estimated Distribution Parameters with 95% Confidence Intervals")
    fig.tight_layout()
    plt.savefig(f"{output_dir}/mean_std_errors.png", dpi=300)
    plt.close()

    # =========================================================================
    # 7. Summary Reporting & Invariant Field Metrics Extraction
    # =========================================================================
    logger.info("\n========== PAPER COMPARISON ==========")
    logger.info("Uniform Q: %.12e", q_uniform)

    fields_path = run_dir / "extracted_fields.npz"
    if fields_path.exists():
        fields_data = np.load(fields_path)
        a_eff, peakiness = compute_field_metrics_from_raw(
            fields_data["vertices_mm"],
            fields_data["triangle_node_indices"],
            fields_data["Jx_nodes"],
            fields_data["Jy_nodes"],
            fields_data["Jz_nodes"]
        )
        logger.info(
            "Effective Area (A_eff): %.5e | Field Peakiness: %.3f",
            a_eff,
            peakiness
        )

    logger.info("Overall Levene p-value: %.6e", levene_p)
    logger.info("Welch ANOVA p-value: %.6e", welch_p)
    logger.info("Saved analysis outputs to: %s", output_dir)
