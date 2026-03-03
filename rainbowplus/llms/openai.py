
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from openai import OpenAI
from rainbowplus.llms.base import BaseLLM
from pydantic import BaseModel
import time
import httpx
import openai

from typing import List


class LLMviaOpenAI(BaseLLM):
    def __init__(self, config):
        self.config = config
        self.model_kwargs = config.model_kwargs
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

        # Check if this is Gemini API and set default safety settings for red-teaming
        self.is_gemini = config.base_url and 'generativelanguage.googleapis.com' in config.base_url
        self.safety_settings = getattr(config, 'safety_settings', None)

        # Default to permissive safety settings for Gemini (useful for red-teaming)
        if self.is_gemini and self.safety_settings is None:
            self.safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

    def get_name(self):
        return self.model_kwargs["model"]

    def generate(self, query: str, sampling_params: dict, max_retries: int = 10, backoff: float = 2.0):
        last_exception = None

        # Note: Gemini's OpenAI-compatible API does NOT support safety_settings
        # The API will return None content when safety filters trigger
        # We handle this by treating None content as a blocked/refused response
        if self.is_gemini and self.safety_settings:
            print(f"WARNING: Gemini OpenAI API does not support safety_settings parameter.")
            print(f"         Safety filters may block responses. Blocked responses will be marked as [BLOCKED].")

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_kwargs["model"],
                    messages=[{"role": "user", "content": query}],
                    **sampling_params
                )

                # Validate response structure
                if response is None:
                    raise ValueError("API returned None response")
                if not response.choices or len(response.choices) == 0:
                    raise ValueError("API returned empty choices list")
                if response.choices[0].message is None:
                    raise ValueError("API returned None message")

                # Log finish_reason for debugging
                choice = response.choices[0]
                finish_reason = getattr(choice, 'finish_reason', None)

                # Check if content is None (could be due to safety filters)
                if choice.message.content is None:
                    # Log detailed information about why content is None
                    print(f"WARNING: API returned None content. finish_reason: {finish_reason}")

                    # Check for safety ratings (Gemini-specific)
                    if hasattr(response, 'prompt_feedback'):
                        print(f"  prompt_feedback: {response.prompt_feedback}")
                    if hasattr(choice, 'safety_ratings'):
                        print(f"  safety_ratings: {choice.safety_ratings}")

                    # If finish_reason indicates safety/content filter, treat as refusal (don't retry)
                    if finish_reason in ['content_filter', 'safety', 'SAFETY', 'RECITATION', 'OTHER']:
                        print(f"  Content blocked by safety filter. Returning refusal message.")
                        return "[BLOCKED: Content filtered by model safety settings]"

                    # Otherwise, treat as an error and retry
                    raise ValueError(f"API returned None content (finish_reason: {finish_reason})")

                return response.choices[0].message.content
            except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError, httpx.HTTPError, httpx.RemoteProtocolError, ValueError) as e:
                last_exception = e
                print(f"OpenAI API error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = backoff * (2 ** attempt)
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)  # Exponential backoff
                else:
                    print(f"All {max_retries} attempts failed. Raising final exception.")
                    raise e
        # If all retries fail, raise the last exception
        raise last_exception

    def batch_generate(self, queries: List[str], sampling_params: dict, delay_between_requests: float = 0.5):
        responses = []
        for i, query in enumerate(queries):
            if i > 0:  # Add delay between requests (not before the first one)
                time.sleep(delay_between_requests)
            responses.append(self.generate(query, sampling_params))
        return responses

    def generate_format(
        self, query: str, sampling_params: dict, response_format: BaseModel
    ):
        completion = self.client.beta.chat.completions.parse(
            model=self.model_kwargs["model"],
            messages=[{"role": "user", "content": query}],
            **sampling_params,
            response_format=response_format
        )
        message = completion.choices[0].message
        return message.parsed