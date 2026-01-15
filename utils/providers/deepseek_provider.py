# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek provider implementation."""

from typing import Any, Dict, List
from .openai_base import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek API provider using OpenAI-compatible interface."""

    def __init__(self):
        super().__init__(
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com"
        )

    @property
    def name(self) -> str:
        return "deepseek"

    def supports_multiple_completions(self) -> bool:
        """DeepSeek currently rejects n>1 for chat.completions.

        The agent will fall back to issuing multiple single-completion calls.
        """
        return False

    def get_max_tokens_limit(self, model_name: str) -> int:
        """Get max tokens limit for DeepSeek models.

        deepseek-reasoner supports up to 65536 tokens.
        deepseek-chat supports up to 8192 tokens.
        """
        if model_name == "deepseek-reasoner":
            return 65536
        return 8192

    def _build_api_params(
        self, model_name: str, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Set temperature to 0.0 for DeepSeek models."""
        api_params = super()._build_api_params(model_name, messages, **kwargs)
        api_params["temperature"] = 0.0
        return api_params
