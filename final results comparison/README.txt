================================================================================
VISUALIZATION GUIDE: Persona-Based Red Teaming Results
================================================================================

This folder contains 12 comprehensive visualizations of your research results.

--------------------------------------------------------------------------------
VISUALIZATION 1: 01_asr_vs_diversity_all.png
--------------------------------------------------------------------------------
WHAT IT SHOWS: Scatter plot of Attack Success Rate vs Prompt Diversity

PURPOSE: Reveals the fundamental trade-off between attack effectiveness and
prompt variety. Ideally, you want methods in the top-right corner.

LOOK FOR:
- Do RedTeaming Expert personas achieve higher ASR but lower diversity?
- Do Regular User personas maintain better diversity?
- Which methods achieve the best balance?

--------------------------------------------------------------------------------
VISUALIZATION 2: 02_improvement_heatmap.png
--------------------------------------------------------------------------------
WHAT IT SHOWS: Heatmap of % improvement over baseline for each method/model

PURPOSE: Shows which persona-based methods improve ASR compared to baseline.
Green = improvement, Red = decline

LOOK FOR:
- Which methods consistently improve across all models?
- Which models are most resistant to persona-based attacks?
- Do RTer personas show higher improvements than User personas?

--------------------------------------------------------------------------------
VISUALIZATION 3: 03_asr_grouped_bars.png
--------------------------------------------------------------------------------
WHAT IT SHOWS: Grouped bar chart comparing absolute ASR values

PURPOSE: Direct comparison of key methods across all target models

LOOK FOR:
- Which model is most vulnerable overall?
- Which method achieves highest ASR for each model?
- Consistency of method performance across models

--------------------------------------------------------------------------------
VISUALIZATION 4: 04_rter_vs_user_comparison.png
--------------------------------------------------------------------------------
WHAT IT SHOWS: 4-panel comparison of RTer vs User personas

PANELS:
- Top-Left: ASR comparison
- Top-Right: Diversity comparison
- Bottom-Left: Dist_Seed (novelty) comparison
- Bottom-Right: Combined scatter plot

PURPOSE: Direct head-to-head comparison of expert vs regular user personas

LOOK FOR:
- Do experts consistently achieve higher ASR?
- Do users compensate with better diversity/novelty?
- Are there models where users outperform experts?

--------------------------------------------------------------------------------
VISUALIZATION 5: 05_three_metrics_comparison.png
--------------------------------------------------------------------------------
WHAT IT SHOWS: 3-panel line plots tracking all three metrics across methods

PANELS:
- Left: ASR progression
- Middle: Diversity progression
- Right: Dist_Seed progression

PURPOSE: Shows how metrics change together or trade off

LOOK FOR:
- Do all metrics improve together or do they trade off?
- Which methods balance all three metrics best?
- How do generated personas (PG) compare to predefined ones?

--------------------------------------------------------------------------------
VISUALIZATIONS 6: 06_detailed_[model].png (5 separate files)
--------------------------------------------------------------------------------
WHAT IT SHOWS: Horizontal bar charts ranking all methods for each model

COLOR CODING:
- Red: RedTeaming Expert personas
- Blue: Regular User personas
- Gray: Baseline and other methods

PURPOSE: Model-specific performance rankings with exact ASR values

LOOK FOR:
- Best method for each specific model
- Variation between methods within a model
- Models with unusual patterns

--------------------------------------------------------------------------------
VISUALIZATION 7: 07_dist_seed_analysis.png
--------------------------------------------------------------------------------
WHAT IT SHOWS: Line plot of Dist_Seed metric across all methods and models

PURPOSE: Tracks prompt novelty/distance from original seeds
Higher = more creative/novel prompts

LOOK FOR:
- Which methods generate the most novel prompts?
- Do personas increase or decrease novelty vs baseline?
- Note unusual low values for Qwen-72B methods

INTERPRETATION:
- High Dist_Seed = More creative prompts, harder to defend
- Low Dist_Seed = Closer to seeds, more predictable

--------------------------------------------------------------------------------
VISUALIZATION 8: 08_summary_statistics.png
--------------------------------------------------------------------------------
WHAT IT SHOWS: Statistical summary table with means and standard deviations

COLOR CODING:
- Light red: RedTeaming Expert methods
- Light blue: Regular User methods
- Gray: Baseline and others

PURPOSE: Overall performance assessment across all models

LOOK FOR:
- Highest mean ASR (best average performance)
- Lowest Std (most consistent across models)
- Methods with high mean + low Std = best overall

INTERPRETATION:
- Mean = Average across all 5 models
- Std = Consistency (Low Std = reliable, High Std = model-dependent)

================================================================================

KEY INSIGHTS TO DRAW FROM YOUR ANALYSIS:

1. EFFECTIVENESS: Which persona type (RTer vs User) achieves higher ASR?

2. DIVERSITY TRADE-OFF: Is there a cost to effectiveness in terms of prompt
   variety and novelty?

3. MODEL ROBUSTNESS: Which models are most/least vulnerable to different
   persona-based attacks?

4. CONSISTENCY: Which methods work reliably across different models?

5. GENERATED vs PREDEFINED: Do generated personas (PG) outperform predefined
   persona sets?

================================================================================
