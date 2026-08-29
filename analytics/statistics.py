"""Statistical analysis of experimental results.

Computes:
  - Mean, std, confidence intervals
  - Percentage improvement
  - Statistical tests (paired t-test, Wilcoxon signed-rank) with assumption checks

Test selection logic:
  - For comparing two related samples (IoT vs traditional):
    * Check normality with Shapiro-Wilk
    * If both normal: paired t-test
    * If not normal: Wilcoxon signed-rank test
  - Documented in results.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List


def compute_summary_stats(values: np.ndarray, confidence: float = 0.95) -> Dict[str, float]:
    """Compute mean, std, and confidence interval."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": 0, "std": 0, "ci_lower": 0, "ci_upper": 0, "n": 0}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    n = len(values)
    if n > 1:
        sem = std / np.sqrt(n)
        t_val = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        ci_lower = mean - t_val * sem
        ci_upper = mean + t_val * sem
    else:
        ci_lower = ci_upper = mean
    return {
        "mean": mean, "std": std,
        "ci_lower": float(ci_lower), "ci_upper": float(ci_upper),
        "n": n,
    }


def check_normality(values: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float]:
    """Shapiro-Wilk normality test. Returns (is_normal, p_value)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 3:
        return True, 1.0  # Assume normal for small samples
    stat, p = stats.shapiro(values)
    return p > alpha, float(p)


def compare_two_groups(group1: np.ndarray, group2: np.ndarray,
                       name: str = "comparison") -> Dict:
    """Compare two related groups with appropriate statistical test.

    Selects paired t-test if both groups are normal, otherwise Wilcoxon.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    min_len = min(len(g1), len(g2))
    g1, g2 = g1[:min_len], g2[:min_len]

    # Remove NaN pairs
    mask = ~np.isnan(g1) & ~np.isnan(g2)
    g1, g2 = g1[mask], g2[mask]

    if len(g1) < 3:
        return {"test": "insufficient_data", "statistic": 0, "p_value": 1.0,
                "significant": False, "name": name}

    norm1, p1 = check_normality(g1)
    norm2, p2 = check_normality(g2)

    if norm1 and norm2:
        stat, p = stats.ttest_rel(g1, g2)
        test_name = "paired_t_test"
    else:
        try:
            stat, p = stats.wilcoxon(g1, g2)
            test_name = "wilcoxon_signed_rank"
        except ValueError:
            stat, p = 0, 1.0
            test_name = "wilcoxon_failed"

    return {
        "test": test_name, "statistic": float(stat), "p_value": float(p),
        "significant": p < 0.05, "name": name,
        "normality_p1": p1, "normality_p2": p2,
    }


def percentage_improvement(iot_val: float, traditional_val: float) -> float:
    """Compute percentage improvement of IoT over traditional.

    Positive = IoT is better. For error metrics, lower is better.
    """
    if traditional_val == 0:
        return 0.0
    return float((traditional_val - iot_val) / traditional_val * 100)


def summarize_experiments(results: Dict[str, List[float]]) -> pd.DataFrame:
    """Summarize repeated experiment results with mean, std, CI."""
    rows = []
    for name, values in results.items():
        s = compute_summary_stats(np.array(values))
        rows.append({
            "metric": name,
            "mean": s["mean"],
            "std": s["std"],
            "ci_lower": s["ci_lower"],
            "ci_upper": s["ci_upper"],
            "n": s["n"],
        })
    return pd.DataFrame(rows)
