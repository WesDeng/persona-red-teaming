data = [

    # --- Table 3: GPT-4o ---
    {'Model': 'GPT-4o', 'Method': 'RP (Baseline)', 'ASR': 0.11, 'Diversity': 0.61, 'Dist_Seed': 1.65},
    {'Model': 'GPT-4o', 'Method': 'RP + RTer0', 'ASR': 0.18, 'Diversity': 0.49, 'Dist_Seed': 1.66},
    {'Model': 'GPT-4o', 'Method': 'RP + RTer1', 'ASR': 0.28, 'Diversity': 0.51, 'Dist_Seed': 1.66},
    {'Model': 'GPT-4o', 'Method': 'RP + User0', 'ASR': 0.13, 'Diversity': 0.6, 'Dist_Seed': 1.85},
    {'Model': 'GPT-4o', 'Method': 'RP + User1', 'ASR': 0.13, 'Diversity': 0.54, 'Dist_Seed': 1.71},
    {'Model': 'GPT-4o', 'Method': 'RP + PG_RTers', 'ASR': 0.23, 'Diversity': 0.62, 'Dist_Seed': 1.72},
    {'Model': 'GPT-4o', 'Method': 'RP + PG_Users', 'ASR': 0.15, 'Diversity': 0.67, 'Dist_Seed': 1.79},
    {'Model': 'GPT-4o', 'Method': 'PG_RTers', 'ASR': 0.16, 'Diversity': 0.63, 'Dist_Seed': 1.73},
    {'Model': 'GPT-4o', 'Method': 'PG_Users', 'ASR': 0.08, 'Diversity': 0.66, 'Dist_Seed': 1.78},

    # --- Table 4: Qwen2.5-7B-Instruct-Turbo ---
    {'Model': 'Qwen-7B', 'Method': 'RP (Baseline)', 'ASR': 0.19, 'Diversity': 0.55, 'Dist_Seed': 1.74, 'ASR_std': 0.0086, 'Div_std': 0.0049},
    {'Model': 'Qwen-7B', 'Method': 'RP + RTer0', 'ASR': 0.26, 'Diversity': 0.46, 'Dist_Seed': 1.65, 'ASR_std': 0.0098, 'Div_std': 0.0045},
    {'Model': 'Qwen-7B', 'Method': 'RP + RTer1', 'ASR': 0.31, 'Diversity': 0.54, 'Dist_Seed': 1.7, 'ASR_std': 0.0104, 'Div_std': 0.005},
    {'Model': 'Qwen-7B', 'Method': 'RP + User0', 'ASR': 0.17, 'Diversity': 0.52, 'Dist_Seed': 1.68, 'ASR_std': 0.0082, 'Div_std': 0.0045},
    {'Model': 'Qwen-7B', 'Method': 'RP + User1', 'ASR': 0.17, 'Diversity': 0.61, 'Dist_Seed': 1.71, 'ASR_std': 0.0084, 'Div_std': 0.0049},
    {'Model': 'Qwen-7B', 'Method': 'RP + PG_RTers', 'ASR': 0.28, 'Diversity': 0.62, 'Dist_Seed': 1.72},
    {'Model': 'Qwen-7B', 'Method': 'RP + PG_Users', 'ASR': 0.19, 'Diversity': 0.68, 'Dist_Seed': 1.77},
    {'Model': 'Qwen-7B', 'Method': 'PG_RTers', 'ASR': 0.21, 'Diversity': 0.54, 'Dist_Seed': 1.69},
    {'Model': 'Qwen-7B', 'Method': 'PG_Users', 'ASR': 0.14, 'Diversity': 0.55, 'Dist_Seed': 1.76},

    # --- Table 5: GPT-4o-mini ---
    {'Model': 'GPT-4o-mini', 'Method': 'RP (Baseline)', 'ASR': 0.15, 'Diversity': 0.6, 'Dist_Seed': 1.7, 'ASR_std': 0.008, 'Div_std': 0.0048},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + RTer0', 'ASR': 0.21, 'Diversity': 0.46, 'Dist_Seed': 1.67, 'ASR_std': 0.0091, 'Div_std': 0.0045},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + RTer1', 'ASR': 0.29, 'Diversity': 0.52, 'Dist_Seed': 1.68, 'ASR_std': 0.0102, 'Div_std': 0.0048},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + User0', 'ASR': 0.14, 'Diversity': 0.5, 'Dist_Seed': 1.65, 'ASR_std': 0.0073, 'Div_std': 0.0044},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + User1', 'ASR': 0.15, 'Diversity': 0.6, 'Dist_Seed': 1.7, 'ASR_std': 0.0081, 'Div_std': 0.005},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + PG_RTers', 'ASR': 0.24, 'Diversity': 0.69, 'Dist_Seed': 1.69},
    {'Model': 'GPT-4o-mini', 'Method': 'RP + PG_Users', 'ASR': 0.19, 'Diversity': 0.78, 'Dist_Seed': 1.74},
    {'Model': 'GPT-4o-mini', 'Method': 'PG_RTers', 'ASR': 0.17, 'Diversity': 0.61, 'Dist_Seed': 1.66},
    {'Model': 'GPT-4o-mini', 'Method': 'PG_Users', 'ASR': 0.1, 'Diversity': 0.66, 'Dist_Seed': 1.69},

    # --- Table 6: Qwen2.5-72B-Instruct-Turbo ---
    {'Model': 'Qwen-72B', 'Method': 'RP (Baseline)', 'ASR': 0.1, 'Diversity': 0.62, 'Dist_Seed': 1.68, 'ASR_std': 0.0067, 'Div_std': 0.0047},
    {'Model': 'Qwen-72B', 'Method': 'RP + RTer0', 'ASR': 0.16, 'Diversity': 0.49, 'Dist_Seed': 1.63, 'ASR_std': 0.0081, 'Div_std': 0.0049},
    {'Model': 'Qwen-72B', 'Method': 'RP + RTer1', 'ASR': 0.23, 'Diversity': 0.54, 'Dist_Seed': 1.7, 'ASR_std': 0.0092, 'Div_std': 0.0048},
    {'Model': 'Qwen-72B', 'Method': 'RP + User0', 'ASR': 0.12, 'Diversity': 0.46, 'Dist_Seed': 1.66, 'ASR_std': 0.0077, 'Div_std': 0.0042},
    {'Model': 'Qwen-72B', 'Method': 'RP + User1', 'ASR': 0.1, 'Diversity': 0.61, 'Dist_Seed': 1.67, 'ASR_std': 0.0069, 'Div_std': 0.0049},
    {'Model': 'Qwen-72B', 'Method': 'RP + PG_RTers', 'ASR': 0.15, 'Diversity': 0.61, 'Dist_Seed': 1.73, 'ASR_std': 0.008, 'Div_std': 0.0051},
    {'Model': 'Qwen-72B', 'Method': 'RP + PG_Users', 'ASR': 0.11, 'Diversity': 0.65, 'Dist_Seed': 1.75, 'ASR_std': 0.007, 'Div_std': 0.0051},
    {'Model': 'Qwen-72B', 'Method': 'PG_RTers', 'ASR': 0.23, 'Diversity': 0.35, 'Dist_Seed': 1.63, 'ASR_std': 0.0094, 'Div_std': 0.0029},
    {'Model': 'Qwen-72B', 'Method': 'PG_Users', 'ASR': 0.21, 'Diversity': 0.1, 'Dist_Seed': 0.84, 'ASR_std': 0.0095, 'Div_std': 0.0021},

    # --- Table 7: Gemini 2.5 Flash ---
    {'Model': 'Gemini Flash', 'Method': 'RP (Baseline)', 'ASR': 0.19, 'Diversity': 0.61, 'Dist_Seed': 1.74, 'ASR_std': 0.0088, 'Div_std': 0.0051},
    {'Model': 'Gemini Flash', 'Method': 'RP + RTer0', 'ASR': 0.23, 'Diversity': 0.5, 'Dist_Seed': 1.63, 'ASR_std': 0.0095, 'Div_std': 0.0047},
    {'Model': 'Gemini Flash', 'Method': 'RP + RTer1', 'ASR': 0.22, 'Diversity': 0.59, 'Dist_Seed': 1.69, 'ASR_std': 0.0089, 'Div_std': 0.0046},
    {'Model': 'Gemini Flash', 'Method': 'RP + User0', 'ASR': 0.16, 'Diversity': 0.57, 'Dist_Seed': 1.67, 'ASR_std': 0.0084, 'Div_std': 0.0045},
    {'Model': 'Gemini Flash', 'Method': 'RP + User1', 'ASR': 0.13, 'Diversity': 0.66, 'Dist_Seed': 1.74, 'ASR_std': 0.0075, 'Div_std': 0.0049},
    {'Model': 'Gemini Flash', 'Method': 'RP + PG_RTers', 'ASR': 0.19, 'Diversity': 0.65, 'Dist_Seed': 1.72, 'ASR_std': 0.0089, 'Div_std': 0.0049},
    {'Model': 'Gemini Flash', 'Method': 'RP + PG_Users', 'ASR': 0.17, 'Diversity': 0.64, 'Dist_Seed': 1.75, 'ASR_std': 0.0087, 'Div_std': 0.0049},
    {'Model': 'Gemini Flash', 'Method': 'PG_RTers', 'ASR': 0.18, 'Diversity': 0.48, 'Dist_Seed': 1.71, 'ASR_std': 0.0085, 'Div_std': 0.0032},
    {'Model': 'Gemini Flash', 'Method': 'PG_Users', 'ASR': 0.47, 'Diversity': 0.09, 'Dist_Seed': 1.0, 'ASR_std': 0.011, 'Div_std': 0.0022},

    # --- Table 8: Gemini 2.5 Pro ---
    {'Model': 'Gemini Pro', 'Method': 'RP (Baseline)', 'ASR': 0.18, 'Diversity': 0.58, 'Dist_Seed': 1.67},
    {'Model': 'Gemini Pro', 'Method': 'RP + RTer0', 'ASR': 0.22, 'Diversity': 0.59, 'Dist_Seed': 1.55},
    {'Model': 'Gemini Pro', 'Method': 'RP + RTer1', 'ASR': 0.21, 'Diversity': 0.56, 'Dist_Seed': 1.64},
    {'Model': 'Gemini Pro', 'Method': 'RP + User0', 'ASR': 0.14, 'Diversity': 0.54, 'Dist_Seed': 1.6},
    {'Model': 'Gemini Pro', 'Method': 'RP + User1', 'ASR': 0.13, 'Diversity': 0.62, 'Dist_Seed': 1.65},
    {'Model': 'Gemini Pro', 'Method': 'RP + PG_RTers', 'ASR': 0.25, 'Diversity': 0.61, 'Dist_Seed': 1.73},
    {'Model': 'Gemini Pro', 'Method': 'RP + PG_Users', 'ASR': 0.15, 'Diversity': 0.6, 'Dist_Seed': 1.77},
    {'Model': 'Gemini Pro', 'Method': 'PG_RTers', 'ASR': 0.19, 'Diversity': 0.45, 'Dist_Seed': 1.62},
    {'Model': 'Gemini Pro', 'Method': 'PG_Users', 'ASR': 0.15, 'Diversity': 0.6, 'Dist_Seed': 0.94},
]
