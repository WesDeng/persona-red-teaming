
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from rainbowplus.llms.base import *
from rainbowplus.llms.openai import *

# Try to import vllm (optional for API-only setups)
try:
    from rainbowplus.llms.vllm import *
except ImportError:
    # vllm not installed, skip vLLM support
    pass

# Try to import Gemini (optional for Google AI setups)
try:
    from rainbowplus.llms.gemini import *
except ImportError:
    # google-generativeai not installed, skip Gemini support
    pass