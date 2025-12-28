
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import google.generativeai as genai
from rainbowplus.llms.base import BaseLLM
import time
from typing import List


class LLMviaGemini(BaseLLM):
    """
    LLM wrapper for Google's Gemini API using the native google-generativeai SDK.
    This implementation properly supports safety_settings configuration, which is
    critical for red-teaming research.
    """

    def __init__(self, config):
        self.config = config
        self.model_kwargs = config.model_kwargs

        # Configure the API with the API key
        genai.configure(api_key=config.api_key)

        # Get safety settings from config or use permissive defaults for red-teaming
        self.safety_settings = getattr(config, 'safety_settings', None)

        # Default to BLOCK_NONE for all categories (essential for red-teaming)
        if self.safety_settings is None:
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                },
            ]

        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=self.model_kwargs["model"],
            safety_settings=self.safety_settings
        )

        print(f"Initialized Gemini model: {self.model_kwargs['model']}")
        print(f"Safety settings: BLOCK_NONE (permissive mode for red-teaming)")

    def get_name(self):
        return self.model_kwargs["model"]

    def generate(self, query: str, sampling_params: dict, max_retries: int = 10, backoff: float = 2.0):
        """
        Generate a response using Gemini's native API.

        Args:
            query: The input prompt
            sampling_params: Sampling parameters (temperature, top_p, max_tokens, etc.)
            max_retries: Maximum number of retry attempts
            backoff: Exponential backoff multiplier

        Returns:
            Generated text response
        """
        last_exception = None

        # Map parameters to Gemini's GenerationConfig
        generation_config = genai.GenerationConfig(
            temperature=sampling_params.get("temperature", 0.7),
            top_p=sampling_params.get("top_p", 0.9),
            max_output_tokens=sampling_params.get("max_tokens", 512),
            # Add other parameters as needed
        )

        for attempt in range(max_retries):
            try:
                # Generate content
                response = self.model.generate_content(
                    query,
                    generation_config=generation_config,
                    safety_settings=self.safety_settings
                )

                # Check if response was blocked by safety filters
                if response.prompt_feedback.block_reason:
                    block_reason = response.prompt_feedback.block_reason
                    print(f"WARNING: Prompt blocked by Gemini safety filters.")
                    print(f"  Block reason: {block_reason}")
                    print(f"  Safety ratings: {response.prompt_feedback.safety_ratings}")
                    return f"[BLOCKED: Prompt blocked - {block_reason}]"

                # Check if any candidate was blocked
                if not response.candidates:
                    print(f"WARNING: No candidates returned (likely blocked).")
                    if hasattr(response, 'prompt_feedback'):
                        print(f"  Prompt feedback: {response.prompt_feedback}")
                    return "[BLOCKED: No candidates returned]"

                # Get the first candidate
                candidate = response.candidates[0]

                # Check finish reason
                finish_reason = candidate.finish_reason

                # If blocked by safety filters, return special marker
                if finish_reason in [3, 4]:  # SAFETY or RECITATION
                    print(f"WARNING: Response blocked by safety filter.")
                    print(f"  Finish reason: {finish_reason}")
                    print(f"  Safety ratings: {candidate.safety_ratings}")
                    return f"[BLOCKED: Content filtered - finish_reason={finish_reason}]"

                # Extract text from response
                if not candidate.content or not candidate.content.parts:
                    print(f"WARNING: Empty content in response.")
                    print(f"  Finish reason: {finish_reason}")
                    return "[BLOCKED: Empty content]"

                # Return the generated text
                return candidate.content.parts[0].text

            except Exception as e:
                last_exception = e
                error_msg = str(e)

                # Check if it's a quota/rate limit error
                if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    print(f"Gemini API rate limit/quota error on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        sleep_time = backoff * (2 ** attempt)
                        print(f"Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        print(f"All {max_retries} attempts failed. Raising final exception.")
                        raise e
                # Check if it's a 503 (service unavailable/overloaded)
                elif "503" in error_msg or "overloaded" in error_msg.lower():
                    print(f"Gemini API overloaded on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        sleep_time = backoff * (2 ** attempt)
                        print(f"Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        print(f"All {max_retries} attempts failed. Raising final exception.")
                        raise e
                else:
                    # For other errors, raise immediately
                    print(f"Gemini API error: {e}")
                    raise e

        # If all retries fail, raise the last exception
        raise last_exception

    def batch_generate(self, queries: List[str], sampling_params: dict, delay_between_requests: float = 0.5):
        """
        Generate responses for a batch of queries.

        Args:
            queries: List of input prompts
            sampling_params: Sampling parameters
            delay_between_requests: Delay between API calls to avoid rate limits

        Returns:
            List of generated responses
        """
        responses = []
        for i, query in enumerate(queries):
            if i > 0:  # Add delay between requests (not before the first one)
                time.sleep(delay_between_requests)
            responses.append(self.generate(query, sampling_params))
        return responses
