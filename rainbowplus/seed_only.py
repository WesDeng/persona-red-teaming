
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import sys
import argparse
import json
import time
import logging
from pathlib import Path

from rainbowplus.scores import OpenAIGuard
from rainbowplus.utils import load_json, initialize_language_models, save_iteration_log
from rainbowplus.archive import Archive
from rainbowplus.configs import ConfigurationLoader
from rainbowplus.prompts import TARGET_PROMPT

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments for seed-only experiment."""
    parser = argparse.ArgumentParser(description="Seed-Only Experiment")
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of seed prompts"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="./configs/seed-only.yml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs-seed-only", help="Directory for storing logs"
    )
    parser.add_argument(
        "--dataset", type=str, default="./data/harmbench.json", help="Dataset path"
    )
    parser.add_argument(
        "--shuffle", type=bool, default=True, help="Shuffle seed prompts"
    )
    return parser.parse_args()


def run_seed_only(args, config, seed_prompts=[], llms=None, fitness_fn=None):
    """Run seed-only experiment without mutations."""
    # Load seed prompts
    if not seed_prompts:
        seed_prompts = load_json(
            config.sample_prompts,
            field="question",
            num_samples=args.num_samples,
            shuffle=args.shuffle,
        )

    # Initialize archives
    adv_prompts = Archive("adv_prompts")
    responses = Archive("responses")
    scores = Archive("scores")
    iters = Archive("iterations")

    # Prepare log directory
    dataset_name = Path(config.sample_prompts).stem
    log_dir = (
        Path(args.log_dir) / config.target_llm.model_kwargs["model"] / dataset_name
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    # Process each seed prompt
    for i, prompt in enumerate(seed_prompts):
        logger.info(f"Processing seed prompt {i+1}/{len(seed_prompts)}")

        # Generate response for seed prompt
        target_prompts = [TARGET_PROMPT.format(prompt=prompt.strip())]
        target_model = config.target_llm.model_kwargs["model"]
        candidates = llms[target_model].batch_generate(
            target_prompts, config.target_llm.sampling_params
        )

        # Score fitness
        fitness_scores = fitness_fn.batch_score(
            [prompt], candidates, config.fitness_llm.sampling_params
        )

        # Create a default key for seed prompts
        key = ("seed", "default")

        # Update archives
        if not adv_prompts.exists(key):
            adv_prompts.add(key, [prompt])
            responses.add(key, candidates)
            scores.add(key, fitness_scores)
            iters.add(key, [0])
        else:
            adv_prompts.extend(key, [prompt])
            responses.extend(key, candidates)
            scores.extend(key, fitness_scores)
            iters.extend(key, [0])

        # Log progress
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {candidates[0]}")
        logger.info(f"Score: {fitness_scores[0]}")
        logger.info("\n\n")

    # Save results
    save_iteration_log(
        log_dir, adv_prompts, responses, scores, iters, "global", iteration=-1
    )
    timestamp = time.strftime(r"%Y%m%d-%H%M%S")
    save_iteration_log(log_dir, adv_prompts, responses, scores, iters, timestamp, iteration=0)

    return adv_prompts, responses, scores


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration
    config = ConfigurationLoader.load(args.config_file)

    # Update configuration based on command-line arguments
    config.sample_prompts = args.dataset

    # Initialize language models and scoring functions
    llms = initialize_language_models(config)
    fitness_fn = OpenAIGuard(config.fitness_llm)

    # Show configuration
    print(config)

    # Run the seed-only experiment
    run_seed_only(
        args,
        config,
        seed_prompts=[],
        llms=llms,
        fitness_fn=fitness_fn,
    ) 