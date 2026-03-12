"""
Comparison graphs:
1. Closed models (GPT-4o, GPT-4o-mini, Gemini Flash, Gemini Pro)
   vs Open models (Qwen-7B, Qwen-72B)
2. Large models (GPT-4o, Qwen-72B, Gemini Pro)
   vs Small models (GPT-4o-mini, Qwen-7B, Gemini Flash)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
data = [
    # GPT-4o
    {'Model': 'GPT-4o', 'Method': 'RP (Baseline)', 'ASR': 0.11, 'Diversity': 0.61},
    {'Model': 'GPT-4o', 'Method': 'RP + RTer0',   'ASR': 0.18, 'Diversity': 0.49},
    {'Model': 'GPT-4o', 'Method': 'RP + RTer1',   'ASR': 0.28, 'Diversity': 0.51},
    {'Model': 'GPT-4o', 'Method': 'RP + User0',   'ASR': 0.13, 'Diversity': 0.60},
    {'Model': 'GPT-4o', 'Method': 'RP + User1',   'ASR': 0.13, 'Diversity': 0.54},
    {'Model': 'GPT-4o', 'Method': 'RP + PG_RTers','ASR': 0.23, 'Diversity': 0.62},
    {'Model': 'GPT-4o', 'Method': 'RP + PG_Users','ASR': 0.15, 'Diversity': 0.67},
    {'Model': 'GPT-4o', 'Method': 'PG_RTers',     'ASR': 0.16, 'Diversity': 0.63},
    {'Model': 'GPT-4o', 'Method': 'PG_Users',     'ASR': 0.08, 'Diversity': 0.66},

    # Qwen-7B
    {'Model': 'Qwen-7B', 'Method': 'RP (Baseline)', 'ASR': 0.19, 'Diversity': 0.533},
    {'Model': 'Qwen-7B', 'Method': 'RP + RTer0',    'ASR': 0.26, 'Diversity': 0.46},
    {'Model': 'Qwen-7B', 'Method': 'RP + RTer1',    'ASR': 0.31, 'Diversity': 0.537},
    {'Model': 'Qwen-7B', 'Method': 'RP + User0',    'ASR': 0.17, 'Diversity': 0.52},
    {'Model': 'Qwen-7B', 'Method': 'RP + User1',    'ASR': 0.16, 'Diversity': 0.61},
    {'Model': 'Qwen-7B', 'Method': 'RP + PG_RTers', 'ASR': 0.28, 'Diversity': 0.62},
    {'Model': 'Qwen-7B', 'Method': 'RP + PG_Users', 'ASR': 0.19, 'Diversity': 0.68},
    {'Model': 'Qwen-7B', 'Method': 'PG_RTers',      'ASR': 0.21, 'Diversity': 0.54},
    {'Model': 'Qwen-7B', 'Method': 'PG_Users',      'ASR': 0.14, 'Diversity': 0.55},

    # GPT-4o-mini
    {'Model': 'GPT-4o-mini', 'Method': 'RP (Baseline)', 'ASR': 0.15, 'Diversity': 0.60},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + RTer0',    'ASR': 0.21, 'Diversity': 0.67},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + RTer1',    'ASR': 0.29, 'Diversity': 0.74},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + User0',    'ASR': 0.14, 'Diversity': 0.70},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + User1',    'ASR': 0.15, 'Diversity': 0.74},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + PG_RTers', 'ASR': 0.24, 'Diversity': 0.69},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + PG_Users', 'ASR': 0.19, 'Diversity': 0.78},
    {'Model': 'GPT-4o-mini', 'Method': 'PG_RTers',      'ASR': 0.17, 'Diversity': 0.61},
    {'Model': 'GPT-4o-mini', 'Method': 'PG_Users',      'ASR': 0.10, 'Diversity': 0.66},

    # Qwen-72B
    {'Model': 'Qwen-72B', 'Method': 'RP (Baseline)', 'ASR': 0.10, 'Diversity': 0.62},
    {'Model': 'Qwen-72B', 'Method': 'RP + RTer0',    'ASR': 0.16, 'Diversity': 0.49},
    {'Model': 'Qwen-72B', 'Method': 'RP + RTer1',    'ASR': 0.23, 'Diversity': 0.54},
    {'Model': 'Qwen-72B', 'Method': 'RP + User0',    'ASR': 0.12, 'Diversity': 0.46},
    {'Model': 'Qwen-72B', 'Method': 'RP + User1',    'ASR': 0.10, 'Diversity': 0.63},
    {'Model': 'Qwen-72B', 'Method': 'RP + PG_RTers', 'ASR': 0.15, 'Diversity': 0.60},
    {'Model': 'Qwen-72B', 'Method': 'RP + PG_Users', 'ASR': 0.11, 'Diversity': 0.65},
    {'Model': 'Qwen-72B', 'Method': 'PG_RTers',      'ASR': 0.23, 'Diversity': 0.35},
    {'Model': 'Qwen-72B', 'Method': 'PG_Users',      'ASR': 0.21, 'Diversity': 0.10},

    # Gemini Flash
    {'Model': 'Gemini Flash', 'Method': 'RP (Baseline)', 'ASR': 0.19, 'Diversity': 0.61},
    {'Model': 'Gemini Flash', 'Method': 'RP + RTer0',    'ASR': 0.23, 'Diversity': 0.62},
    {'Model': 'Gemini Flash', 'Method': 'RP + RTer1',    'ASR': 0.22, 'Diversity': 0.59},
    {'Model': 'Gemini Flash', 'Method': 'RP + User0',    'ASR': 0.16, 'Diversity': 0.57},
    {'Model': 'Gemini Flash', 'Method': 'RP + User1',    'ASR': 0.13, 'Diversity': 0.66},
    {'Model': 'Gemini Flash', 'Method': 'RP + PG_RTers', 'ASR': 0.19, 'Diversity': 0.65},
    {'Model': 'Gemini Flash', 'Method': 'RP + PG_Users', 'ASR': 0.17, 'Diversity': 0.64},
    {'Model': 'Gemini Flash', 'Method': 'PG_RTers',      'ASR': 0.18, 'Diversity': 0.48},
    {'Model': 'Gemini Flash', 'Method': 'PG_Users',      'ASR': 0.17, 'Diversity': 0.64},

    # Gemini Pro
    {'Model': 'Gemini Pro', 'Method': 'RP (Baseline)', 'ASR': 0.18, 'Diversity': 0.58},
    {'Model': 'Gemini Pro', 'Method': 'RP + RTer0',    'ASR': 0.22, 'Diversity': 0.59},
    {'Model': 'Gemini Pro', 'Method': 'RP + RTer1',    'ASR': 0.21, 'Diversity': 0.56},
    {'Model': 'Gemini Pro', 'Method': 'RP + User0',    'ASR': 0.14, 'Diversity': 0.54},
    {'Model': 'Gemini Pro', 'Method': 'RP + User1',    'ASR': 0.13, 'Diversity': 0.62},
    {'Model': 'Gemini Pro', 'Method': 'RP + PG_RTers', 'ASR': 0.25, 'Diversity': 0.61},
    {'Model': 'Gemini Pro', 'Method': 'RP + PG_Users', 'ASR': 0.15, 'Diversity': 0.60},
    {'Model': 'Gemini Pro', 'Method': 'PG_RTers',      'ASR': 0.19, 'Diversity': 0.45},
    {'Model': 'Gemini Pro', 'Method': 'PG_Users',      'ASR': 0.15, 'Diversity': 0.60},
]

df = pd.DataFrame(data)

# ── Group labels ──────────────────────────────────────────────────────────────
CLOSED_MODELS = {'GPT-4o', 'GPT-4o-mini', 'Gemini Flash', 'Gemini Pro'}
OPEN_MODELS   = {'Qwen-7B', 'Qwen-72B'}
LARGE_MODELS  = {'GPT-4o', 'Qwen-72B', 'Gemini Pro'}
SMALL_MODELS  = {'GPT-4o-mini', 'Qwen-7B', 'Gemini Flash'}

df['ModelType'] = df['Model'].apply(lambda m: 'Closed' if m in CLOSED_MODELS else 'Open')
df['ModelSize'] = df['Model'].apply(lambda m: 'Large' if m in LARGE_MODELS else 'Small')

METHOD_ORDER = [
    'RP (Baseline)',
    'RP + RTer1',
    'RP + User1',
    'RP + PG_RTers',
    'RP + PG_Users',
]

output_dir = Path('final results comparison')
output_dir.mkdir(exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
import seaborn as sns
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'sans-serif'

COLOR_A = '#2196F3'   # blue  (closed / large)
COLOR_B = '#FF5722'   # orange (open / small)
ALPHA   = 0.82
WIDTH   = 0.35


def grouped_bar_comparison(group_col, label_a, label_b, color_a, color_b,
                            title_prefix, out_filename):
    """
    For each method plot two bars: group A (averaged across its models)
    vs group B (averaged across its models). Two panels: ASR and Diversity.
    """
    means_a = (df[df[group_col] == label_a]
               .groupby('Method')[['ASR', 'Diversity']].mean())
    means_b = (df[df[group_col] == label_b]
               .groupby('Method')[['ASR', 'Diversity']].mean())

    # Reindex to fixed method order
    means_a = means_a.reindex(METHOD_ORDER)
    means_b = means_b.reindex(METHOD_ORDER)

    x = np.arange(len(METHOD_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{title_prefix}: {label_a} vs {label_b} Models',
                 fontsize=16, fontweight='bold', y=1.01)

    for ax, metric, ylim in zip(axes, ['ASR', 'Diversity'], [0.35, 0.85]):
        bars_a = ax.bar(x - WIDTH / 2, means_a[metric], WIDTH,
                        label=label_a, color=color_a, alpha=ALPHA,
                        edgecolor='white', linewidth=0.6)
        bars_b = ax.bar(x + WIDTH / 2, means_b[metric], WIDTH,
                        label=label_b, color=color_b, alpha=ALPHA,
                        edgecolor='white', linewidth=0.6)

        # Value labels
        for bar in list(bars_a) + list(bars_b):
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7.5)

        ax.set_title(metric, fontsize=14)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, rotation=40, ha='right', fontsize=9)
        ax.set_ylim(0, ylim)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / out_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_dir / out_filename}')


# ── Graph 1: Closed vs Open ───────────────────────────────────────────────────
grouped_bar_comparison(
    group_col='ModelType',
    label_a='Closed', label_b='Open',
    color_a=COLOR_A, color_b=COLOR_B,
    title_prefix='Closed (GPT, Gemini) vs Open (Qwen)',
    out_filename='10_closed_vs_open_models.png',
)

# ── Graph 2: Large vs Small ───────────────────────────────────────────────────
grouped_bar_comparison(
    group_col='ModelSize',
    label_a='Large', label_b='Small',
    color_a='#4CAF50', color_b='#9C27B0',
    title_prefix='Large (GPT-4o, Qwen-72B, Gemini Pro) vs Small (GPT-4o-mini, Qwen-7B, Gemini Flash)',
    out_filename='11_large_vs_small_models.png',
)
