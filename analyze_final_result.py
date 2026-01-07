"""
================================================================================
PERSONA-BASED RED TEAMING: COMPREHENSIVE RESULTS VISUALIZATION
================================================================================

RESEARCH OVERVIEW:
This script visualizes results from a persona-based red teaming study that
compares different strategies for generating adversarial prompts to test AI
safety across multiple language models.

METHODS COMPARED:
1. RP (Baseline) - Rainbow Teaming with risk categories and attack styles
2. RP + RTer0/1 - Rainbow Teaming + RedTeaming Expert personas (2 variants)
3. RP + User0/1 - Rainbow Teaming + Regular AI User personas (2 variants)
4. RP + PG_RTers - Rainbow Teaming + Generated RedTeaming Expert personas
5. RP + PG_Users - Rainbow Teaming + Generated Regular User personas
6. PG_RTers - Generated RedTeaming Expert personas only
7. PG_Users - Generated Regular User personas only

TARGET MODELS:
- GPT-4o
- Qwen2.5-7B (Qwen-7B)
- GPT-4o-mini
- Qwen2.5-72B (Qwen-72B)
- Gemini 2.5 Flash

METRICS:
- ASR (Attack Success Rate): Percentage of successful jailbreak attempts
- Diversity: Variety/uniqueness of generated prompts (higher is better)
- Dist_Seed: Distance from original seed prompts (novelty measure)

OUTPUTS:
This script generates 12 high-resolution visualizations in the
'final results comparison/' directory:
01. ASR vs Diversity trade-off scatter plot
02. Improvement heatmap (% change from baseline)
03. Grouped bar chart of ASR across models
04. RTer vs User 4-panel comparison
05. Three metrics progression line plots
06. Per-model detailed rankings (5 separate charts)
07. Distance from seed analysis
08. Summary statistics table

All visualizations are saved at 300 DPI for publication quality.

================================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# 1. DATA INJECTION
# ---------------------------------------------------------
# Data extracted from LaTeX tables (Tables 3-7) in the research paper.
# Each row represents one method tested on one target model.
# ---------------------------------------------------------

data = [
    # --- Table 3: GPT-4o ---
    {'Model': 'GPT-4o', 'Method': 'RP (Baseline)', 'ASR': 0.11, 'Diversity': 0.61, 'Dist_Seed': 1.65},
    {'Model': 'GPT-4o', 'Method': 'RP + RTer0',   'ASR': 0.18, 'Diversity': 0.49, 'Dist_Seed': 1.66},
    {'Model': 'GPT-4o', 'Method': 'RP + RTer1',   'ASR': 0.28, 'Diversity': 0.51, 'Dist_Seed': 1.66},
    {'Model': 'GPT-4o', 'Method': 'RP + User0',   'ASR': 0.13, 'Diversity': 0.60, 'Dist_Seed': 1.85},
    {'Model': 'GPT-4o', 'Method': 'RP + User1',   'ASR': 0.13, 'Diversity': 0.54, 'Dist_Seed': 1.71},
    {'Model': 'GPT-4o', 'Method': 'RP + PG_RTers','ASR': 0.23, 'Diversity': 0.62, 'Dist_Seed': 1.72},
    {'Model': 'GPT-4o', 'Method': 'RP + PG_Users','ASR': 0.15, 'Diversity': 0.67, 'Dist_Seed': 1.79},
    {'Model': 'GPT-4o', 'Method': 'PG_RTers',     'ASR': 0.16, 'Diversity': 0.63, 'Dist_Seed': 1.73},
    {'Model': 'GPT-4o', 'Method': 'PG_Users',     'ASR': 0.08, 'Diversity': 0.66, 'Dist_Seed': 1.78},

    # --- Table 4: Qwen2.5-7B ---
    {'Model': 'Qwen-7B', 'Method': 'RP (Baseline)', 'ASR': 0.19, 'Diversity': 0.533, 'Dist_Seed': 1.30},
    {'Model': 'Qwen-7B', 'Method': 'RP + RTer0',    'ASR': 0.26, 'Diversity': 0.46,  'Dist_Seed': 1.30},
    {'Model': 'Qwen-7B', 'Method': 'RP + RTer1',    'ASR': 0.31, 'Diversity': 0.537, 'Dist_Seed': 1.31},
    {'Model': 'Qwen-7B', 'Method': 'RP + User0',    'ASR': 0.17, 'Diversity': 0.52,  'Dist_Seed': 1.317},
    {'Model': 'Qwen-7B', 'Method': 'RP + User1',    'ASR': 0.16, 'Diversity': 0.61,  'Dist_Seed': 1.29},
    {'Model': 'Qwen-7B', 'Method': 'RP + PG_RTers', 'ASR': 0.28, 'Diversity': 0.62,  'Dist_Seed': 1.33},
    {'Model': 'Qwen-7B', 'Method': 'RP + PG_Users', 'ASR': 0.19, 'Diversity': 0.68,  'Dist_Seed': 1.29},
    {'Model': 'Qwen-7B', 'Method': 'PG_RTers',      'ASR': 0.21, 'Diversity': 0.54,  'Dist_Seed': 1.27},
    {'Model': 'Qwen-7B', 'Method': 'PG_Users',      'ASR': 0.14, 'Diversity': 0.55,  'Dist_Seed': 1.31},

    # --- Table 5: GPT-4o-mini ---
    {'Model': 'GPT-4o-mini', 'Method': 'RP (Baseline)', 'ASR': 0.15, 'Diversity': 0.60, 'Dist_Seed': 1.30},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + RTer0',    'ASR': 0.21, 'Diversity': 0.67, 'Dist_Seed': 1.30},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + RTer1',    'ASR': 0.29, 'Diversity': 0.74, 'Dist_Seed': 1.32},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + User0',    'ASR': 0.14, 'Diversity': 0.70, 'Dist_Seed': 1.31},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + User1',    'ASR': 0.15, 'Diversity': 0.74, 'Dist_Seed': 1.34},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + PG_Users', 'ASR': 0.19, 'Diversity': 0.78, 'Dist_Seed': 1.39},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + PG_RTers', 'ASR': 0.24, 'Diversity': 0.69, 'Dist_Seed': 1.35},
    {'Model': 'GPT-4o-mini', 'Method': 'PG_RTers',      'ASR': 0.17, 'Diversity': 0.61, 'Dist_Seed': 1.23},
    {'Model': 'GPT-4o-mini', 'Method': 'PG_Users',      'ASR': 0.10, 'Diversity': 0.66, 'Dist_Seed': 1.39},

    # --- Table 6: Qwen2.5-72B ---
    {'Model': 'Qwen-72B', 'Method': 'RP (Baseline)', 'ASR': 0.10, 'Diversity': 0.62, 'Dist_Seed': 1.28},
    {'Model': 'Qwen-72B', 'Method': 'RP + RTer0',    'ASR': 0.16, 'Diversity': 0.49, 'Dist_Seed': 1.30},
    {'Model': 'Qwen-72B', 'Method': 'RP + RTer1',    'ASR': 0.23, 'Diversity': 0.54, 'Dist_Seed': 1.32},
    {'Model': 'Qwen-72B', 'Method': 'RP + User0',    'ASR': 0.12, 'Diversity': 0.46, 'Dist_Seed': 1.32},
    {'Model': 'Qwen-72B', 'Method': 'RP + User1',    'ASR': 0.10, 'Diversity': 0.63, 'Dist_Seed': 1.31},
    {'Model': 'Qwen-72B', 'Method': 'RP + PG_RTers', 'ASR': 0.15, 'Diversity': 0.60, 'Dist_Seed': 0.67},
    {'Model': 'Qwen-72B', 'Method': 'RP + PG_Users', 'ASR': 0.11, 'Diversity': 0.65, 'Dist_Seed': 0.67},
    {'Model': 'Qwen-72B', 'Method': 'PG_RTers',      'ASR': 0.23, 'Diversity': 0.35, 'Dist_Seed': 0.44},
    {'Model': 'Qwen-72B', 'Method': 'PG_Users',      'ASR': 0.21, 'Diversity': 0.10, 'Dist_Seed': 0.26},

    # --- Table 7: Gemini 2.5 flash ---
    {'Model': 'Gemini Flash', 'Method': 'RP (Baseline)', 'ASR': 0.19, 'Diversity': 0.61, 'Dist_Seed': 1.28},
    {'Model': 'Gemini Flash', 'Method': 'RP + RTer0',    'ASR': 0.23, 'Diversity': 0.62, 'Dist_Seed': 1.12},
    {'Model': 'Gemini Flash', 'Method': 'RP + RTer1',    'ASR': 0.22, 'Diversity': 0.59, 'Dist_Seed': 1.31},
    {'Model': 'Gemini Flash', 'Method': 'RP + User0',    'ASR': 0.16, 'Diversity': 0.57, 'Dist_Seed': 1.31},
    {'Model': 'Gemini Flash', 'Method': 'RP + User1',    'ASR': 0.13, 'Diversity': 0.66, 'Dist_Seed': 1.31},
    {'Model': 'Gemini Flash', 'Method': 'RP + PG_RTers', 'ASR': 0.19, 'Diversity': 0.65, 'Dist_Seed': 1.28},
    {'Model': 'Gemini Flash', 'Method': 'RP + PG_Users', 'ASR': 0.17, 'Diversity': 0.64, 'Dist_Seed': 1.30},
    {'Model': 'Gemini Flash', 'Method': 'PG_RTers',      'ASR': 0.18, 'Diversity': 0.48, 'Dist_Seed': 1.29},
    {'Model': 'Gemini Flash', 'Method': 'PG_Users',      'ASR': 0.17, 'Diversity': 0.64, 'Dist_Seed': 1.30},
]

df = pd.DataFrame(data)

# Create output directory
output_dir = Path('final results comparison')
output_dir.mkdir(exist_ok=True)

# Calculate Relative Improvement over Baseline for each Model
baseline_asrs = df[df['Method'] == 'RP (Baseline)'].set_index('Model')['ASR']
def calc_improvement(row):
    base = baseline_asrs[row['Model']]
    return ((row['ASR'] - base) / base) * 100
df['Improvement_Pct'] = df.apply(calc_improvement, axis=1)

# Add persona type classification
def classify_persona(method):
    if 'RTer' in method:
        return 'RedTeaming Experts'
    elif 'User' in method:
        return 'Regular Users'
    elif 'Baseline' in method:
        return 'Baseline'
    else:
        return 'Other'

df['PersonaType'] = df['Method'].apply(classify_persona)

# Set global style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150

# ---------------------------------------------------------
# VISUALIZATION 1: ASR vs Diversity Trade-off (All Methods)
# ---------------------------------------------------------
# PURPOSE: This scatter plot reveals the fundamental trade-off between attack
# success and prompt diversity. Ideally, you want methods in the top-right
# corner (high ASR and high diversity).
#
# INSIGHTS TO LOOK FOR:
# - Do RedTeaming Expert personas achieve higher ASR but lower diversity?
# - Do Regular User personas maintain better diversity while sacrificing some ASR?
# - Are there methods that achieve both high ASR and high diversity?
# - How does this trade-off vary across different target models?
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))
plot_df = df.copy()

for persona_type in plot_df['PersonaType'].unique():
    subset = plot_df[plot_df['PersonaType'] == persona_type]
    for model in subset['Model'].unique():
        model_subset = subset[subset['Model'] == model]
        ax.scatter(model_subset['Diversity'], model_subset['ASR'],
                  label=f'{persona_type} ({model})', s=150, alpha=0.7)

ax.set_title('Trade-off: Attack Success Rate vs. Prompt Diversity (All Methods)', fontsize=16, pad=20)
ax.set_xlabel('Diversity Score', fontsize=13)
ax.set_ylabel('Attack Success Rate (ASR)', fontsize=13)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '01_asr_vs_diversity_all.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# VISUALIZATION 2: Relative Improvement Heatmap
# ---------------------------------------------------------
# PURPOSE: This heatmap shows the percentage improvement (or decline) in ASR
# compared to the baseline RP method for each target model. Green indicates
# improvement, red indicates decline.
#
# INSIGHTS TO LOOK FOR:
# - Which persona-based methods consistently improve ASR across all models?
# - Are there models that are more resistant to persona-based attacks?
# - Do RTer personas show higher improvement percentages than User personas?
# - Are there any methods that perform worse than baseline (red cells)?
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))
heatmap_data = df.pivot(index='Method', columns='Model', values='Improvement_Pct')
heatmap_data = heatmap_data.drop('RP (Baseline)')

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn",
    center=0,
    cbar_kws={'label': '% Improvement over Baseline'},
    linewidths=0.5,
    ax=ax
)
ax.set_title('Percentage Improvement in ASR over Baseline', fontsize=16, pad=20)
ax.set_xlabel('Target Model', fontsize=13)
ax.set_ylabel('Attack Method', fontsize=13)
plt.tight_layout()
plt.savefig(output_dir / '02_improvement_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# VISUALIZATION 3: Grouped Bar Chart - ASR across Models
# ---------------------------------------------------------
# PURPOSE: This grouped bar chart allows direct comparison of absolute ASR values
# for key methods across all target models. This shows which combinations of
# method + model achieve the highest attack success rates.
#
# INSIGHTS TO LOOK FOR:
# - Which model is most vulnerable to attacks (tallest bars overall)?
# - Which method achieves the highest ASR for each model?
# - How consistent is each method's performance across different models?
# - Is there a clear winner among RTer1, User1, and generated persona methods?
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))
method_order = ['RP (Baseline)', 'RP + RTer1', 'RP + User1', 'RP + PG_RTers', 'RP + PG_Users']
filtered_df = df[df['Method'].isin(method_order)]

x = np.arange(len(filtered_df['Model'].unique()))
width = 0.15
methods = filtered_df['Method'].unique()

for i, method in enumerate(method_order):
    method_data = filtered_df[filtered_df['Method'] == method]
    offset = (i - len(method_order)/2) * width
    ax.bar(x + offset, method_data['ASR'], width, label=method, alpha=0.8)

ax.set_title('Attack Success Rate Comparison Across Target Models', fontsize=16, pad=20)
ax.set_xlabel('Target Model', fontsize=13)
ax.set_ylabel('Attack Success Rate (ASR)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(filtered_df['Model'].unique())
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / '03_asr_grouped_bars.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# VISUALIZATION 4: RTer vs User Persona Comparison
# ---------------------------------------------------------
# PURPOSE: This 4-panel comparison directly contrasts RedTeaming Expert personas
# against Regular User personas across all three metrics. This is crucial for
# understanding whether "expert" personas offer advantages over regular users.
#
# PANEL 4a (Top-Left): ASR Comparison - Do experts achieve higher attack success?
# PANEL 4b (Top-Right): Diversity Comparison - Do users maintain better diversity?
# PANEL 4c (Bottom-Left): Dist_Seed Comparison - Who generates more novel prompts?
# PANEL 4d (Bottom-Right): Combined Scatter - Overall trade-off visualization
#
# INSIGHTS TO LOOK FOR:
# - Is there a consistent advantage for RTer personas in ASR?
# - Do User personas compensate with higher diversity or novelty?
# - Are there models where User personas actually outperform RTer personas?
# - What's the optimal persona type for balancing all three metrics?
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 4a: ASR comparison
ax = axes[0, 0]
rter_df = df[df['PersonaType'] == 'RedTeaming Experts']
user_df = df[df['PersonaType'] == 'Regular Users']

x = np.arange(len(df['Model'].unique()))
width = 0.35

rter_means = rter_df.groupby('Model')['ASR'].mean()
user_means = user_df.groupby('Model')['ASR'].mean()

ax.bar(x - width/2, rter_means, width, label='RedTeaming Experts', alpha=0.8, color='#e74c3c')
ax.bar(x + width/2, user_means, width, label='Regular Users', alpha=0.8, color='#3498db')

ax.set_title('ASR: RedTeaming Experts vs Regular Users', fontsize=14, pad=15)
ax.set_xlabel('Target Model', fontsize=12)
ax.set_ylabel('Average ASR', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(rter_means.index, rotation=15)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4b: Diversity comparison
ax = axes[0, 1]
rter_div = rter_df.groupby('Model')['Diversity'].mean()
user_div = user_df.groupby('Model')['Diversity'].mean()

ax.bar(x - width/2, rter_div, width, label='RedTeaming Experts', alpha=0.8, color='#e74c3c')
ax.bar(x + width/2, user_div, width, label='Regular Users', alpha=0.8, color='#3498db')

ax.set_title('Diversity: RedTeaming Experts vs Regular Users', fontsize=14, pad=15)
ax.set_xlabel('Target Model', fontsize=12)
ax.set_ylabel('Average Diversity', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(rter_div.index, rotation=15)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4c: Dist_Seed comparison
ax = axes[1, 0]
rter_dist = rter_df.groupby('Model')['Dist_Seed'].mean()
user_dist = user_df.groupby('Model')['Dist_Seed'].mean()

ax.bar(x - width/2, rter_dist, width, label='RedTeaming Experts', alpha=0.8, color='#e74c3c')
ax.bar(x + width/2, user_dist, width, label='Regular Users', alpha=0.8, color='#3498db')

ax.set_title('Distance from Seed: RedTeaming Experts vs Regular Users', fontsize=14, pad=15)
ax.set_xlabel('Target Model', fontsize=12)
ax.set_ylabel('Average Dist_Seed', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(rter_dist.index, rotation=15)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4d: Combined metric scatter
ax = axes[1, 1]
for model in df['Model'].unique():
    model_rter = rter_df[rter_df['Model'] == model]
    model_user = user_df[user_df['Model'] == model]

    if not model_rter.empty:
        ax.scatter(model_rter['Diversity'].mean(), model_rter['ASR'].mean(),
                  s=200, marker='o', label=f'{model} (RTer)', alpha=0.7)
    if not model_user.empty:
        ax.scatter(model_user['Diversity'].mean(), model_user['ASR'].mean(),
                  s=200, marker='s', label=f'{model} (User)', alpha=0.7)

ax.set_title('ASR vs Diversity: Persona Type Comparison', fontsize=14, pad=15)
ax.set_xlabel('Average Diversity', fontsize=12)
ax.set_ylabel('Average ASR', fontsize=12)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '04_rter_vs_user_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# VISUALIZATION 5: Three Metrics Comparison (Line Plots)
# ---------------------------------------------------------
# PURPOSE: This 3-panel line plot tracks how ASR, Diversity, and Dist_Seed
# change across different methods for each model. This helps identify which
# metrics improve together and which trade off against each other.
#
# LEFT PANEL: ASR progression - shows attack effectiveness across methods
# MIDDLE PANEL: Diversity progression - shows prompt variety maintenance
# RIGHT PANEL: Dist_Seed progression - shows novelty/distance from original seeds
#
# INSIGHTS TO LOOK FOR:
# - Do all three metrics move in the same direction or do they trade off?
# - Which methods achieve the best balance across all three metrics?
# - Are there models where adding personas hurts some metrics?
# - How do generated personas (PG) compare to predefined personas?
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

models = df['Model'].unique()
top_methods = ['RP (Baseline)', 'RP + RTer1', 'RP + PG_RTers', 'RP + PG_Users']

for idx, metric in enumerate(['ASR', 'Diversity', 'Dist_Seed']):
    ax = axes[idx]

    for model in models:
        model_df = df[(df['Model'] == model) & (df['Method'].isin(top_methods))]
        model_df = model_df.set_index('Method')

        x_pos = np.arange(len(top_methods))
        values = [model_df.loc[m, metric] if m in model_df.index else 0 for m in top_methods]

        ax.plot(x_pos, values, marker='o', label=model, linewidth=2, markersize=8)

    ax.set_title(f'{metric} Across Methods', fontsize=14, pad=15)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_methods, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '05_three_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# VISUALIZATION 6: Per-Model Detailed Comparison (5 separate charts)
# ---------------------------------------------------------
# PURPOSE: These horizontal bar charts provide a detailed, model-specific ranking
# of all attack methods. One chart is generated for each target model, showing
# which methods work best against that specific model.
#
# COLOR CODING:
# - Red bars: RedTeaming Expert personas
# - Blue bars: Regular User personas
# - Gray bars: Baseline and other methods
#
# INSIGHTS TO LOOK FOR:
# - What's the best performing method for each specific model?
# - How much variation is there between methods for a given model?
# - Are certain models more resistant to all types of attacks?
# - Do some models show unusual patterns (e.g., User personas outperform RTer)?
# - Which methods consistently rank high across multiple models?
# ---------------------------------------------------------
for model in df['Model'].unique():
    fig, ax = plt.subplots(figsize=(12, 8))

    model_df = df[df['Model'] == model].sort_values('ASR', ascending=True)

    y_pos = np.arange(len(model_df))
    colors = ['#e74c3c' if 'RTer' in m else '#3498db' if 'User' in m else '#95a5a6'
              for m in model_df['Method']]

    bars = ax.barh(y_pos, model_df['ASR'], color=colors, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_df['Method'])
    ax.set_xlabel('Attack Success Rate (ASR)', fontsize=12)
    ax.set_title(f'Attack Success Rate for {model}', fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, model_df['ASR'])):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
               f'{val:.2f}', va='center', fontsize=10)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.7, label='RedTeaming Experts'),
        Patch(facecolor='#3498db', alpha=0.7, label='Regular Users'),
        Patch(facecolor='#95a5a6', alpha=0.7, label='Baseline/Other')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    safe_model_name = model.replace(' ', '_').replace('.', '_')
    plt.savefig(output_dir / f'06_detailed_{safe_model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# VISUALIZATION 7: Distance from Seed Analysis
# ---------------------------------------------------------
# PURPOSE: This line plot tracks the Dist_Seed metric (distance from original
# seed prompts) across all methods and models. Higher values indicate more novel
# prompts that diverge further from the original seeds. This metric was UNUSED
# in the original visualization but is crucial for understanding prompt novelty.
#
# INSIGHTS TO LOOK FOR:
# - Which methods generate the most novel prompts (highest Dist_Seed)?
# - Do persona-based methods increase or decrease novelty compared to baseline?
# - Are there models where certain methods fail to diversify (low Dist_Seed)?
# - Note: Qwen-72B shows unusual low values for some methods - investigate why
# - Do generated personas (PG) produce more novel prompts than predefined ones?
#
# INTERPRETATION:
# - High Dist_Seed = More creative/novel prompts, potentially harder to defend
# - Low Dist_Seed = Prompts closer to seeds, potentially more predictable
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))

for method in df['Method'].unique():
    method_df = df[df['Method'] == method]
    ax.plot(method_df['Model'], method_df['Dist_Seed'],
           marker='o', label=method, linewidth=2, markersize=8)

ax.set_title('Distance from Seed Prompts Across Models and Methods', fontsize=16, pad=20)
ax.set_xlabel('Target Model', fontsize=13)
ax.set_ylabel('Dist_Seed', fontsize=13)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(output_dir / '07_dist_seed_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# VISUALIZATION 8: Summary Statistics Table
# ---------------------------------------------------------
# PURPOSE: This table provides a comprehensive statistical summary of all methods
# across all target models, showing both mean values and standard deviations for
# each metric. This helps assess consistency and variability of each method.
#
# COLOR CODING:
# - Light red background: RedTeaming Expert methods
# - Light blue background: Regular User methods
# - Gray background: Baseline and other methods
#
# INSIGHTS TO LOOK FOR:
# - Which method has the highest mean ASR across all models?
# - Which method shows the most consistent performance (lowest Std)?
# - Are RTer or User personas more stable across different models?
# - High Std values indicate method performance varies significantly by model
# - Look for methods with high mean and low Std (best overall performers)
#
# INTERPRETATION:
# - Mean: Average performance across all 5 target models
# - Std (Standard Deviation): Consistency of performance
#   * Low Std = Consistent across models (reliable)
#   * High Std = Variable across models (model-dependent)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

summary_data = []
for method in df['Method'].unique():
    method_df = df[df['Method'] == method]
    summary_data.append([
        method,
        f"{method_df['ASR'].mean():.3f}",
        f"{method_df['ASR'].std():.3f}",
        f"{method_df['Diversity'].mean():.3f}",
        f"{method_df['Diversity'].std():.3f}",
        f"{method_df['Dist_Seed'].mean():.3f}",
        f"{method_df['Dist_Seed'].std():.3f}"
    ])

table = ax.table(cellText=summary_data,
                colLabels=['Method', 'ASR Mean', 'ASR Std', 'Diversity Mean', 'Diversity Std',
                          'Dist_Seed Mean', 'Dist_Seed Std'],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color code the cells
for i in range(len(summary_data)):
    if 'RTer' in summary_data[i][0]:
        color = '#ffe6e6'
    elif 'User' in summary_data[i][0]:
        color = '#e6f2ff'
    else:
        color = '#f0f0f0'

    for j in range(7):
        table[(i+1, j)].set_facecolor(color)

plt.title('Summary Statistics: Mean and Standard Deviation Across Models',
         fontsize=16, pad=20, y=0.98)
plt.savefig(output_dir / '08_summary_statistics.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll visualizations saved to '{output_dir}' directory!")
print(f"Generated {len(list(output_dir.glob('*.png')))} visualization files.")