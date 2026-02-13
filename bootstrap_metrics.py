#!/usr/bin/env python3
"""
Bootstrap Standard Deviations for ASR and Diversity.

- ASR: bootstrapped from per-prompt binary outcomes in comprehensive_log_global.json
- Diversity (1 - Self-BLEU): point estimate from summary.json;
  SD by computing per-prompt BLEU scores once (O(N^2)), then bootstrapping
  by resampling those scores (O(N) per replicate).

Usage:
    python bootstrap_metrics.py
    python bootstrap_metrics.py --bootstrap-replicates 5000
"""

import json
import argparse
import os
import numpy as np
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ---------------------------------------------------------------------------
# Experiment directory mapping
# ---------------------------------------------------------------------------

# Maps (Model, Method) -> experiment directory name under exp_results/
EXPERIMENT_MAP = {
    # --- Gemini Flash ---
    ("Gemini Flash", "RP (Baseline)"): "aws-results-exp-gemini-flash-RP-baseline-20251221-110317",
    ("Gemini Flash", "RP + RTer0"):    "aws-results-exp-gemini-flash-RP-RTer0-20251225-053426",
    ("Gemini Flash", "RP + RTer1"):    "aws-results-exp-gemini-flash-RP-RTer1-20251220-012158",
    ("Gemini Flash", "RP + User0"):    "aws-results-exp-gemini-flash-RP-User0-20251221-022317",
    ("Gemini Flash", "RP + User1"):    "aws-results-exp-gemini-flash-RP-User1-20251221-024029",
    ("Gemini Flash", "RP + PG_RTers"): "aws-results-exp-gemini-flash-RP-PG-Rter-20251226-032222",
    ("Gemini Flash", "RP + PG_Users"): "aws-results-exp-gemini-flash-RP-PG-User-20251226-142157",
    ("Gemini Flash", "PG_RTers"):      "aws-results-exp-gemini-flash-PG-Rter-20251227-034717",
    ("Gemini Flash", "PG_Users"):      "aws-results-exp-gemini-flash-PG-User-20251228-033117",

    # --- GPT-4o-mini ---
    ("GPT-4o-mini", "RP (Baseline)"): "aws-results-exp-gpt-4o-mini-RP-baseline-20251216-225552",
    ("GPT-4o-mini", "RP + RTer0"):    "aws-results-exp-gpt-4o-mini-RP-RTer0-20251215-004005",
    ("GPT-4o-mini", "RP + RTer1"):    "aws-results-exp-gpt-4o-mini-RP-RTer1-20251215-110041",
    ("GPT-4o-mini", "RP + User0"):    "aws-results-exp-gpt-4o-mini-RP-User0-20251215-204622",
    ("GPT-4o-mini", "RP + User1"):    "aws-results-exp-gpt-4o-mini-RP-User1-20251216-052129",

    # --- Qwen-72B ---
    ("Qwen-72B", "RP (Baseline)"): "aws-results-exp-qwen-RP-baseline-20251218-135903",
    ("Qwen-72B", "RP + RTer0"):    "aws-results-exp-qwen-RP-RTer0-20251216-235052",
    ("Qwen-72B", "RP + RTer1"):    "aws-results-exp-qwen-RP-RTer1-20251217-093446",
    ("Qwen-72B", "RP + User0"):    "aws-results-exp-qwen-RP-User0-20251217-181308",
    ("Qwen-72B", "RP + User1"):    "aws-results-exp-qwen-RP-User1-20251218-034814",
    ("Qwen-72B", "RP + PG_RTers"): "aws-results-exp-Qwen-72B-RP-PG-Rter-20251230-135047",
    ("Qwen-72B", "RP + PG_Users"): "aws-results-exp-Qwen-72B-RP-PG-User-20251230-012443",
    ("Qwen-72B", "PG_RTers"):      "aws-results-exp-Qwen-72B-PG-Rter-20251228-202143",
    ("Qwen-72B", "PG_Users"):       "aws-results-exp-Qwen-72B-PG-User-20251230-223236",

    # --- Qwen-7B ---
    ("Qwen-7B", "RP (Baseline)"): "logs-qwen-RP-baseline",
    ("Qwen-7B", "RP + RTer0"):    "logs-qwen-RP-RTer0",
    ("Qwen-7B", "RP + RTer1"):    "logs-qwen-RP-RTer1",
    ("Qwen-7B", "RP + User0"):    "logs-qwen-RP-User0",
    ("Qwen-7B", "RP + User1"):    "logs-qwen-RP-User1",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def find_file_recursive(base_dir, filename):
    """Find a file recursively under base_dir."""
    for root, _, files in os.walk(base_dir):
        if filename in files:
            return str(Path(root) / filename)
    return None


def load_experiment_data(log_path):
    """Load prompts and binary acceptance outcomes from comprehensive_log_global.json."""
    with open(log_path) as f:
        data = json.load(f)

    all_prompts = data.get("all_prompts", {})
    rejection_reasons = data.get("rejection_reasons", {})

    flat_prompts = []
    flat_reasons = []
    for key, prompts in all_prompts.items():
        flat_prompts.extend(prompts)
        flat_reasons.extend(rejection_reasons.get(key, ["unknown"] * len(prompts)))

    accepted = np.array([1 if r == "accepted" else 0 for r in flat_reasons])
    return flat_prompts, accepted


def load_diversity_from_summary(summary_path):
    """Read self_bleu from summary.json and return diversity = 1 - self_bleu."""
    with open(summary_path) as f:
        summary = json.load(f)

    self_bleu = summary["diversity"]["self_bleu"]
    return 1.0 - self_bleu, self_bleu


# ---------------------------------------------------------------------------
# Per-prompt Self-BLEU (computed once, then resampled for bootstrap)
# ---------------------------------------------------------------------------


def compute_per_prompt_bleu(flat_prompts, n_gram=4):
    """
    Compute per-prompt Self-BLEU scores.

    For each prompt i, BLEU(prompt_i, references=all_other_prompts).
    Returns array of N scores whose mean = corpus Self-BLEU.
    This is the expensive O(N^2) step, done once per experiment.
    """
    tokenized = [p.split() for p in flat_prompts]
    smoother = SmoothingFunction().method1
    weights = tuple([1 / n_gram] * n_gram)
    n = len(tokenized)
    bleu_scores = np.zeros(n)

    for i, candidate in enumerate(tokenized):
        references = tokenized[:i] + tokenized[i + 1:]
        bleu_scores[i] = sentence_bleu(
            references, candidate, weights=weights, smoothing_function=smoother
        )
        if (i + 1) % 500 == 0:
            print(f"      BLEU progress: {i+1}/{n}")

    return bleu_scores


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_proportion(binary_outcomes, n_replicates=1000, seed=42):
    """Bootstrap SD and 95% CI for a proportion (e.g. ASR)."""
    rng = np.random.RandomState(seed)
    n = len(binary_outcomes)
    point_est = float(binary_outcomes.mean())

    boot_means = np.zeros(n_replicates)
    for i in range(n_replicates):
        idx = rng.choice(n, size=n, replace=True)
        boot_means[i] = binary_outcomes[idx].mean()

    return {
        "point_estimate": point_est,
        "std": float(boot_means.std()),
        "ci_95_lower": float(np.percentile(boot_means, 2.5)),
        "ci_95_upper": float(np.percentile(boot_means, 97.5)),
        "n_replicates": n_replicates,
        "n_samples": n,
    }


def bootstrap_diversity(bleu_scores, diversity_point_est, n_replicates=1000, seed=42):
    """
    Bootstrap SD for diversity = 1 - Self-BLEU.

    Resamples N per-prompt BLEU_i scores with replacement; the mean of
    each resample gives a simulated Self-BLEU, and 1 - that is diversity.
    Point estimate comes from summary.json (pre-computed self_bleu).
    """
    rng = np.random.RandomState(seed)
    n = len(bleu_scores)

    boot_diversities = np.zeros(n_replicates)
    for i in range(n_replicates):
        idx = rng.choice(n, size=n, replace=True)
        boot_diversities[i] = 1.0 - bleu_scores[idx].mean()

    return {
        "point_estimate": diversity_point_est,
        "self_bleu": 1.0 - diversity_point_est,
        "std": float(boot_diversities.std()),
        "ci_95_lower": float(np.percentile(boot_diversities, 2.5)),
        "ci_95_upper": float(np.percentile(boot_diversities, 97.5)),
        "n_replicates": n_replicates,
        "n_samples": n,
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_all_experiments(exp_results_root, n_replicates=1000, seed=42):
    """Process all mapped experiments, returning results keyed by (Model, Method)."""
    results = {}

    total = len(EXPERIMENT_MAP)
    for i, ((model, method), exp_dir_name) in enumerate(sorted(EXPERIMENT_MAP.items())):
        exp_path = Path(exp_results_root) / exp_dir_name
        label = f"{model} / {method}"

        if not exp_path.is_dir():
            print(f"  [{i+1}/{total}] SKIP (dir not found): {label}")
            continue

        print(f"  [{i+1}/{total}] Processing: {label}")

        # --- Load prompts + acceptance from comprehensive_log ---
        log_path = find_file_recursive(exp_path, "comprehensive_log_global.json")
        if not log_path:
            print(f"    WARNING: No comprehensive_log_global.json found, skipping")
            results[(model, method)] = {
                "model": model, "method": method, "exp_dir": exp_dir_name,
                "ASR": None, "diversity": None,
            }
            continue

        flat_prompts, accepted = load_experiment_data(log_path)
        n = len(flat_prompts)
        print(f"    {n} total samples")

        # --- ASR bootstrap ---
        print(f"    Bootstrapping ASR...")
        asr_result = bootstrap_proportion(accepted, n_replicates=n_replicates, seed=seed)

        # --- Diversity: point estimate from summary.json ---
        summary_path = find_file_recursive(exp_path, "summary.json")
        if summary_path and "attack_analysis" in summary_path:
            diversity_pt, self_bleu = load_diversity_from_summary(summary_path)
        else:
            print(f"    WARNING: No attack_analysis/summary.json, skipping diversity")
            results[(model, method)] = {
                "model": model, "method": method, "exp_dir": exp_dir_name,
                "ASR": asr_result, "diversity": None,
            }
            continue

        # --- Diversity bootstrap: compute per-prompt BLEU once, then resample ---
        print(f"    Computing per-prompt BLEU for {n} prompts (one-time O(N^2))...")
        bleu_scores = compute_per_prompt_bleu(flat_prompts)
        print(f"    Bootstrapping diversity (1 - Self-BLEU)...")
        div_result = bootstrap_diversity(
            bleu_scores, diversity_pt, n_replicates=n_replicates, seed=seed
        )

        results[(model, method)] = {
            "model": model,
            "method": method,
            "exp_dir": exp_dir_name,
            "ASR": asr_result,
            "diversity": div_result,
        }

    return results


def print_summary(results):
    """Print formatted summary table."""
    print("\n" + "=" * 120)
    print("BOOTSTRAP RESULTS: ASR + Diversity (1 - Self-BLEU)")
    print("=" * 120)
    print(f"{'Model':<15} {'Method':<18} {'ASR':>26} {'Diversity (1-BLEU)':>28}")
    print("-" * 120)

    current_model = None
    for (model, method), res in sorted(results.items()):
        if model != current_model:
            if current_model is not None:
                print()
            current_model = model

        asr = res["ASR"]
        div = res["diversity"]

        if asr:
            asr_str = f"{asr['point_estimate']:.4f} +/- {asr['std']:.4f}"
        else:
            asr_str = "N/A"

        if isinstance(div, dict):
            div_str = f"{div['point_estimate']:.4f} +/- {div['std']:.4f}"
        elif div is not None:
            div_str = f"{div:.4f}"
        else:
            div_str = "N/A"

        print(f"{model:<15} {method:<18} {asr_str:>26} {div_str:>28}")


def generate_updated_table(results):
    """Generate the updated data table (Python dict format) for analyze_final_result.py."""
    lines = []
    lines.append("data = [")

    # Group by model
    models_order = ["GPT-4o", "Qwen-7B", "GPT-4o-mini", "Qwen-72B", "Gemini Flash", "Gemini Pro"]
    methods_order = [
        "RP (Baseline)", "RP + RTer0", "RP + RTer1", "RP + User0", "RP + User1",
        "RP + PG_RTers", "RP + PG_Users", "PG_RTers", "PG_Users",
    ]

    # Original data for experiments we don't have local data for (ASR, Diversity, Dist_Seed)
    original_data = {
        ("GPT-4o", "RP (Baseline)"): (0.11, 0.61, 1.65),
        ("GPT-4o", "RP + RTer0"):    (0.18, 0.49, 1.66),
        ("GPT-4o", "RP + RTer1"):    (0.28, 0.51, 1.66),
        ("GPT-4o", "RP + User0"):    (0.13, 0.60, 1.85),
        ("GPT-4o", "RP + User1"):    (0.13, 0.54, 1.71),
        ("GPT-4o", "RP + PG_RTers"): (0.23, 0.62, 1.72),
        ("GPT-4o", "RP + PG_Users"): (0.15, 0.67, 1.79),
        ("GPT-4o", "PG_RTers"):      (0.16, 0.63, 1.73),
        ("GPT-4o", "PG_Users"):      (0.08, 0.66, 1.78),

        ("Gemini Pro", "RP (Baseline)"): (0.18, 0.58, 1.67),
        ("Gemini Pro", "RP + RTer0"):    (0.22, 0.59, 1.55),
        ("Gemini Pro", "RP + RTer1"):    (0.21, 0.56, 1.64),
        ("Gemini Pro", "RP + User0"):    (0.14, 0.54, 1.60),
        ("Gemini Pro", "RP + User1"):    (0.13, 0.62, 1.65),
        ("Gemini Pro", "RP + PG_RTers"): (0.25, 0.61, 1.73),
        ("Gemini Pro", "RP + PG_Users"): (0.15, 0.60, 1.77),
        ("Gemini Pro", "PG_RTers"):      (0.19, 0.45, 1.62),
        ("Gemini Pro", "PG_Users"):      (0.15, 0.60, 0.94),

        # GPT-4o-mini PG experiments (no local data)
        ("GPT-4o-mini", "RP + PG_RTers"): (0.24, 0.69, 1.69),
        ("GPT-4o-mini", "RP + PG_Users"): (0.19, 0.78, 1.74),
        ("GPT-4o-mini", "PG_RTers"):      (0.17, 0.61, 1.66),
        ("GPT-4o-mini", "PG_Users"):      (0.10, 0.66, 1.69),

        # Qwen-7B PG experiments (no local data)
        ("Qwen-7B", "RP + PG_RTers"): (0.28, 0.62, 1.72),
        ("Qwen-7B", "RP + PG_Users"): (0.19, 0.68, 1.77),
        ("Qwen-7B", "PG_RTers"):      (0.21, 0.54, 1.69),
        ("Qwen-7B", "PG_Users"):      (0.14, 0.55, 1.76),
    }

    # Original Dist_Seed values (not recomputed here)
    dist_seed_data = {
        ("GPT-4o", "RP (Baseline)"): 1.65, ("GPT-4o", "RP + RTer0"): 1.66,
        ("GPT-4o", "RP + RTer1"): 1.66, ("GPT-4o", "RP + User0"): 1.85,
        ("GPT-4o", "RP + User1"): 1.71, ("GPT-4o", "RP + PG_RTers"): 1.72,
        ("GPT-4o", "RP + PG_Users"): 1.79, ("GPT-4o", "PG_RTers"): 1.73,
        ("GPT-4o", "PG_Users"): 1.78,

        ("Qwen-7B", "RP (Baseline)"): 1.74, ("Qwen-7B", "RP + RTer0"): 1.65,
        ("Qwen-7B", "RP + RTer1"): 1.70, ("Qwen-7B", "RP + User0"): 1.68,
        ("Qwen-7B", "RP + User1"): 1.71, ("Qwen-7B", "RP + PG_RTers"): 1.72,
        ("Qwen-7B", "RP + PG_Users"): 1.77, ("Qwen-7B", "PG_RTers"): 1.69,
        ("Qwen-7B", "PG_Users"): 1.76,

        ("GPT-4o-mini", "RP (Baseline)"): 1.70, ("GPT-4o-mini", "RP + RTer0"): 1.67,
        ("GPT-4o-mini", "RP + RTer1"): 1.68, ("GPT-4o-mini", "RP + User0"): 1.65,
        ("GPT-4o-mini", "RP + User1"): 1.70, ("GPT-4o-mini", "RP + PG_RTers"): 1.69,
        ("GPT-4o-mini", "RP + PG_Users"): 1.74, ("GPT-4o-mini", "PG_RTers"): 1.66,
        ("GPT-4o-mini", "PG_Users"): 1.69,

        ("Qwen-72B", "RP (Baseline)"): 1.68, ("Qwen-72B", "RP + RTer0"): 1.63,
        ("Qwen-72B", "RP + RTer1"): 1.70, ("Qwen-72B", "RP + User0"): 1.66,
        ("Qwen-72B", "RP + User1"): 1.67, ("Qwen-72B", "RP + PG_RTers"): 1.73,
        ("Qwen-72B", "RP + PG_Users"): 1.75, ("Qwen-72B", "PG_RTers"): 1.63,
        ("Qwen-72B", "PG_Users"): 0.84,

        ("Gemini Flash", "RP (Baseline)"): 1.74, ("Gemini Flash", "RP + RTer0"): 1.63,
        ("Gemini Flash", "RP + RTer1"): 1.69, ("Gemini Flash", "RP + User0"): 1.67,
        ("Gemini Flash", "RP + User1"): 1.74, ("Gemini Flash", "RP + PG_RTers"): 1.72,
        ("Gemini Flash", "RP + PG_Users"): 1.75, ("Gemini Flash", "PG_RTers"): 1.71,
        ("Gemini Flash", "PG_Users"): 1.00,

        ("Gemini Pro", "RP (Baseline)"): 1.67, ("Gemini Pro", "RP + RTer0"): 1.55,
        ("Gemini Pro", "RP + RTer1"): 1.64, ("Gemini Pro", "RP + User0"): 1.60,
        ("Gemini Pro", "RP + User1"): 1.65, ("Gemini Pro", "RP + PG_RTers"): 1.73,
        ("Gemini Pro", "RP + PG_Users"): 1.77, ("Gemini Pro", "PG_RTers"): 1.62,
        ("Gemini Pro", "PG_Users"): 0.94,
    }

    model_comments = {
        "GPT-4o": "Table 3: GPT-4o",
        "Qwen-7B": "Table 4: Qwen2.5-7B-Instruct-Turbo",
        "GPT-4o-mini": "Table 5: GPT-4o-mini",
        "Qwen-72B": "Table 6: Qwen2.5-72B-Instruct-Turbo",
        "Gemini Flash": "Table 7: Gemini 2.5 Flash",
        "Gemini Pro": "Table 8: Gemini 2.5 Pro",
    }

    for model in models_order:
        lines.append(f"\n    # --- {model_comments[model]} ---")
        for method in methods_order:
            key = (model, method)

            # Get ASR and Diversity from bootstrap results or original data
            if key in results and results[key]["ASR"] is not None:
                asr = round(results[key]["ASR"]["point_estimate"], 2)
                asr_std = round(results[key]["ASR"]["std"], 4)
            elif key in original_data:
                asr = original_data[key][0]
                asr_std = None
            else:
                continue

            div_res = results.get(key, {}).get("diversity")
            if div_res is not None and isinstance(div_res, dict):
                diversity = round(div_res["point_estimate"], 2)
                div_std = round(div_res["std"], 4)
            elif div_res is not None:
                diversity = round(div_res, 2)
                div_std = None
            elif key in original_data:
                diversity = original_data[key][1]
                div_std = None
            else:
                continue

            dist_seed = dist_seed_data.get(key, 0.0)

            # Format entry
            std_parts = []
            if asr_std is not None:
                std_parts.append(f"'ASR_std': {asr_std}")
            if div_std is not None:
                std_parts.append(f"'Div_std': {div_std}")
            std_str = ", " + ", ".join(std_parts) if std_parts else ""
            lines.append(
                f"    {{'Model': '{model}', 'Method': '{method}', "
                f"'ASR': {asr}, 'Diversity': {diversity}, 'Dist_Seed': {dist_seed}{std_str}}},"
            )

    lines.append("]")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compute bootstrap SDs for ASR; read diversity from summary.json (1 - self_bleu)"
    )
    parser.add_argument(
        "--exp-results-root",
        default="exp_results",
        help="Root directory containing experiment results (default: exp_results)",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
        help="Number of bootstrap replicates (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output",
        default="bootstrap_results.json",
        help="Output JSON file (default: bootstrap_results.json)",
    )
    args = parser.parse_args()

    print(f"Bootstrap replicates: {args.bootstrap_replicates}")
    print(f"Experiments mapped: {len(EXPERIMENT_MAP)}")
    print()

    # Process all experiments
    results = process_all_experiments(
        args.exp_results_root,
        n_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )

    # Print summary
    print_summary(results)

    # Save detailed results as JSON
    json_results = {}
    for (model, method), res in sorted(results.items()):
        json_key = f"{model} / {method}"
        json_results[json_key] = res
    with open(args.output, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nDetailed results saved to {args.output}")

    # Generate and save updated table
    table_str = generate_updated_table(results)
    table_output = args.output.replace(".json", "_table.py")
    with open(table_output, "w") as f:
        f.write(table_str + "\n")
    print(f"Updated data table saved to {table_output}")


if __name__ == "__main__":
    main()
