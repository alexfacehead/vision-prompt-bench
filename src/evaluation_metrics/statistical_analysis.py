import logging
import numpy as np
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)


def analyze_metric(base_values: list[float], improved_values: list[float],
                   metric_name: str, higher_is_better: bool) -> dict:
    """
    Run a paired Wilcoxon signed-rank test on a single metric.

    Args:
        base_values: Metric values from base prompt images.
        improved_values: Metric values from enhanced prompt images.
        metric_name: Name of the metric.
        higher_is_better: If True, improvement means improved > base.

    Returns:
        Dict with test results including p-value, effect size, CI.
    """
    base = np.array(base_values, dtype=float)
    improved = np.array(improved_values, dtype=float)

    if higher_is_better:
        diffs = improved - base
        direction = "higher"
    else:
        diffs = base - improved  # positive diff = improvement
        direction = "lower"

    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0

    result = {
        "metric": metric_name,
        "n": n,
        "direction": direction,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "base_mean": float(np.mean(base)),
        "improved_mean": float(np.mean(improved)),
    }

    # Cohen's d effect size
    if std_diff > 0:
        result["cohens_d"] = mean_diff / std_diff
    else:
        result["cohens_d"] = 0.0

    # 95% confidence interval
    if n > 1:
        se = std_diff / np.sqrt(n)
        result["ci_95"] = (mean_diff - 1.96 * se, mean_diff + 1.96 * se)
    else:
        result["ci_95"] = (mean_diff, mean_diff)

    # Wilcoxon signed-rank test (requires n >= 6 for meaningful results)
    if n >= 6:
        try:
            non_zero_diffs = diffs[diffs != 0]
            if len(non_zero_diffs) >= 1:
                stat, p_value = wilcoxon(non_zero_diffs, alternative="greater")
                result["wilcoxon_stat"] = float(stat)
                result["p_value"] = float(p_value)
                result["significant"] = p_value < 0.05
            else:
                result["p_value"] = 1.0
                result["significant"] = False
                result["note"] = "All differences are zero"
        except ValueError as e:
            logger.warning("Wilcoxon test failed for %s: %s", metric_name, e)
            result["p_value"] = None
            result["significant"] = None
    else:
        result["p_value"] = None
        result["significant"] = None
        result["note"] = f"Sample size too small (n={n}) for Wilcoxon test; need >= 6"

    return result


def run_statistical_analysis(base_quality_list: list[dict],
                             improved_quality_list: list[dict]) -> list[dict]:
    """
    Run statistical tests on all no-reference quality metrics.

    Args:
        base_quality_list: List of quality metric dicts for base images.
        improved_quality_list: List of quality metric dicts for improved images.

    Returns:
        List of analysis result dicts, one per metric.
    """
    metrics_config = {
        "brisque": {"higher_is_better": False},  # lower BRISQUE = better quality
        "entropy": {"higher_is_better": True},   # higher entropy = more information
        "colorfulness": {"higher_is_better": True},  # higher = more colorful
    }

    results = []
    for metric_name, config in metrics_config.items():
        base_values = [q[metric_name] for q in base_quality_list]
        improved_values = [q[metric_name] for q in improved_quality_list]

        analysis = analyze_metric(
            base_values, improved_values,
            metric_name, config["higher_is_better"]
        )
        results.append(analysis)

    return results


def format_report(analysis_results: list[dict],
                  similarity_results: list[dict] = None) -> str:
    """Format statistical analysis results as a readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("PROMPT OPTIMIZATION EVALUATION REPORT")
    lines.append("=" * 70)

    lines.append("\n--- Quality Metrics (No-Reference) ---")
    lines.append("These measure absolute image quality. Improvement means the")
    lines.append("enhanced prompt produced higher-quality images.\n")

    for r in analysis_results:
        lines.append(f"  {r['metric'].upper()}")
        lines.append(f"    Base mean:     {r['base_mean']:.4f}")
        lines.append(f"    Improved mean: {r['improved_mean']:.4f}")
        lines.append(f"    Mean diff:     {r['mean_diff']:+.4f} ({r['direction']} is better)")
        lines.append(f"    Cohen's d:     {r['cohens_d']:.4f}")

        ci = r["ci_95"]
        lines.append(f"    95% CI:        ({ci[0]:+.4f}, {ci[1]:+.4f})")

        if r.get("p_value") is not None:
            sig = "YES" if r["significant"] else "no"
            lines.append(f"    p-value:       {r['p_value']:.6f} (significant: {sig})")
        elif r.get("note"):
            lines.append(f"    Note:          {r['note']}")

        # Interpret effect size
        d = abs(r["cohens_d"])
        if d >= 0.8:
            effect = "large"
        elif d >= 0.5:
            effect = "medium"
        elif d >= 0.2:
            effect = "small"
        else:
            effect = "negligible"
        lines.append(f"    Effect size:   {effect}")
        lines.append("")

    if similarity_results:
        lines.append("--- Similarity Metrics (Reference-Based) ---")
        lines.append("These show how different the improved images are from base images.")
        lines.append("They do NOT measure quality, only difference.\n")

        for key, values in similarity_results.items():
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0
            lines.append(f"  {key.upper()}: {mean_val:.4f} (+/- {std_val:.4f})")
        lines.append("")

    # Overall conclusion
    lines.append("--- Conclusion ---")
    sig_improvements = [r for r in analysis_results
                        if r.get("significant") is True]
    if sig_improvements:
        names = ", ".join(r["metric"].upper() for r in sig_improvements)
        lines.append(f"  Statistically significant improvement in: {names}")
    else:
        has_small_n = any(r.get("note", "").startswith("Sample") for r in analysis_results)
        if has_small_n:
            lines.append("  Insufficient sample size for statistical significance testing.")
            lines.append("  Generate more images (>= 6 per prompt) for Wilcoxon test.")
        else:
            lines.append("  No statistically significant improvements detected.")

    lines.append("=" * 70)
    return "\n".join(lines)
