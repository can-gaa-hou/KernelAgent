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

import os
import logging
from pathlib import Path
from typing import Any, Dict, List
from .openai_base import LLMResponse, OpenAICompatibleProvider

from ..tools import CodeSearcher, CommandRunner, FileEditor


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek API provider using OpenAI-compatible interface."""

    def __init__(self):
        super().__init__(
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com"
        )

        # Build tools if provider is available
        self.tools = self.build_tools() if int(os.getenv("TOOL_CALLS", "0")) else None

        if self.tools:
            self.files = FileEditor()
            self.searcher = CodeSearcher()
            self.runner = CommandRunner()

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
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> dict[str, Any]:
        """Build API parameters for OpenAI-compatible call."""
        # Add system prompt if provided
        if "system_prompt" in kwargs and kwargs["system_prompt"]:
            messages = [{"role": "system", "content": kwargs["system_prompt"]}] + messages

        params = {
            "model": model_name,
            "messages": messages,
        }
        
        params["temperature"] = kwargs.get("temperature", 0.0)

        params["max_tokens"] = min(
            kwargs.get("max_tokens", 8192), self.get_max_tokens_limit(model_name)
        )

        # Add n parameter if specified
        if "n" in kwargs:
            params["n"] = kwargs["n"]
        
        # Add tools if provided
        if "tools" in kwargs and kwargs["tools"]:
            params["tools"] = kwargs["tools"]
            params["tool_choice"] = "auto"

        if "stream" in kwargs and kwargs["stream"]:
            params["stream"] = kwargs["stream"]
            params["stream_options"] = {"include_usage": True}

        return params

    def get_response(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Get single response."""
        if not self.is_available():
            raise RuntimeError(f"{self.name} client not available")
        
        if "tools" in kwargs and kwargs["tools"]:
            kwargs["tools"] = self.tools

        api_params = self._build_api_params(model_name, messages, **kwargs)
        response = self.client.chat.completions.create(**api_params)

        logging.getLogger(__name__).info(
            "DeepSeek chat response (single): %s",
            getattr(response, "model_dump", lambda: str(response))(),
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=model_name,
            provider=self.name,
            usage=response.usage.dict()
            if hasattr(response, "usage") and response.usage
            else None,
            tool_calls=[tc.model_dump() for tc in response.choices[0].message.tool_calls] if response.choices[0].message.tool_calls else None,
        )

    def _stream_response(self, response):
        for chunk in response:
            if not chunk.choices:
                continue
                
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason
            
            yield_data = delta.model_dump(exclude_unset=True)
            if finish_reason:
                yield_data["finish_reason"] = finish_reason
            
            yield yield_data
    
    def build_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Create or overwrite file content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "replace_in_file",
                    "description": "Replace a single unique occurrence of a string in a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old_str": {"type": "string", "description": "The string to search for. Must be unique in the file."},
                            "new_str": {"type": "string", "description": "The replacement string."}
                        },
                        "required": ["path", "old_str", "new_str"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "background": {"type": "boolean"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search for files or code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"}
                        },
                        "required": ["pattern"]
                    }
                }
            }
        ]

    def execute_tool(self, name: str, args: Dict[str, Any], workdir: Path) -> str:
        try:
            if name == "read_file":
                path = (workdir / args["path"]).resolve()
                return self.files.read_file(str(path))
            elif name == "write_file":
                path = (workdir / args["path"]).resolve()
                return self.files.create_file(str(path), args["content"])
            elif name == "replace_in_file":
                path = (workdir / args["path"]).resolve()
                return self.files.replace_in_file(str(path), args["old_str"], args["new_str"])
            elif name == "run_command":
                if args.get("background"):
                    handle = self.runner.run_nonblocking(args["command"], str(workdir))
                    return f"Started background command with PID {handle.pid}"
                else:
                    res = self.runner.run(args["command"], str(workdir))
                    return f"Exit: {res.returncode}\nOutput: {res.stdout}\nError: {res.stderr}"
            elif name == "search_files":
                # Determine if it is file search or code search based on pattern
                # For simplicity, map to code search
                hits = self.searcher.search(args["pattern"])
                return "\n".join([f"{f}:{l} {c.strip()}" for f, l, c in hits[:20]]) # Limit results
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Tool execution error: {str(e)}"
