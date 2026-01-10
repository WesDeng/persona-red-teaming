import json
import os
from pathlib import Path

# Find all summary.json files
summary_files = []
for root, dirs, files in os.walk("exp_results"):
    for file in files:
        if file == "summary.json":
            summary_files.append(os.path.join(root, file))

# Extract data
results = []
for file_path in summary_files:
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract experiment name from path
    path_parts = file_path.split('/')
    exp_name = path_parts[1]  # The experiment directory name

    # Extract metrics
    avg_nu = data.get('avg_nu_pairwise_L2_distances', 'N/A')
    std_nu = data.get('std_nu_pairwise_L2_distances', 'N/A')
    avg_sp = data.get('avg_sp_pairwise_L2_distances', 'N/A')
    std_sp = data.get('std_sp_pairwise_L2_distances', 'N/A')

    # Format the metrics
    if avg_nu != 'N/A' and std_nu != 'N/A':
        distance_nearest = f"${avg_nu:.4f} \\pm {std_nu:.4f}$"
    else:
        distance_nearest = "N/A"

    if avg_sp != 'N/A' and std_sp != 'N/A':
        distance_seed = f"${avg_sp:.4f} \\pm {std_sp:.4f}$"
    else:
        distance_seed = "N/A"

    results.append({
        'experiment': exp_name,
        'distance_nearest': distance_nearest,
        'distance_seed': distance_seed,
        'avg_nu': avg_nu,
        'avg_sp': avg_sp
    })

# Sort results by experiment name
results.sort(key=lambda x: x['experiment'])

# Print as markdown table
print("| Experiment | Distance_Nearest | Distance_Seed |")
print("|------------|------------------|---------------|")
for r in results:
    print(f"| {r['experiment']} | {r['distance_nearest']} | {r['distance_seed']} |")

print("\n\n")
print("=" * 80)
print("Summary Statistics:")
print("=" * 80)

# Group by model and condition
from collections import defaultdict
grouped = defaultdict(list)

for r in results:
    exp = r['experiment']
    if 'gemini' in exp.lower():
        model = 'Gemini-2.5-Flash'
    elif 'gpt-4o-mini' in exp.lower():
        model = 'GPT-4o-mini'
    elif 'qwen' in exp.lower() or 'Qwen-72B' in exp:
        if 'Qwen2.5-7B' in exp or 'logs-qwen' in exp:
            model = 'Qwen-2.5-7B'
        else:
            model = 'Qwen-2.5-72B'
    else:
        model = 'Unknown'

    # Determine condition
    if 'baseline' in exp.lower():
        condition = 'Baseline'
    elif 'RTer0' in exp or 'Rter0' in exp:
        condition = 'RTer0'
    elif 'RTer1' in exp or 'Rter1' in exp:
        condition = 'RTer1'
    elif 'User0' in exp:
        condition = 'User0'
    elif 'User1' in exp:
        condition = 'User1'
    elif 'PG-User' in exp or 'PG_User' in exp:
        condition = 'PG-User'
    elif 'PG-Rter' in exp or 'PG_Rter' in exp or 'PG-RTer' in exp:
        condition = 'PG-Rter'
    else:
        condition = 'Other'

    grouped[(model, condition)].append(r)

print("\nGrouped by Model and Condition:")
print("-" * 80)
for (model, condition), items in sorted(grouped.items()):
    print(f"\n{model} - {condition}:")
    for item in items:
        print(f"  {item['experiment']}")
        print(f"    Distance_Nearest: {item['distance_nearest']}")
        print(f"    Distance_Seed: {item['distance_seed']}")
