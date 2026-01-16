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

"""Context manager for maintaining conversation history."""

from typing import List, Dict, Any, Optional
try:
    import tiktoken
except ImportError:
    tiktoken = None

class ContextManager:
    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        self.encoding = None
        if tiktoken:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            
        self.messages: List[Dict[str, Any]] = []
        self.system_prompt: str = ""
        self.max_context_tokens = 128000 # Default for deepseek-chat
        
    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def add_message(self, role: str, content: str, name: str = None, tool_calls: List = None, tool_call_id: str = None):
        msg = {"role": role}
        if content is not None:
            msg["content"] = content
        if name:
            msg["name"] = name
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
            
        self.messages.append(msg)

    def get_messages(self) -> List[Dict[str, Any]]:
        # Implement sliding window logic if needed
        # For now, just return all, maybe checking limit
        current_tokens = self.count_tokens(self.messages)
        if current_tokens > self.max_context_tokens:
            # Simple truncation: keep system prompt + last N messages
            # This is a naive implementation. A better one would summarize or drop middle.
            # But "1:1" usually means "smart context management". 
            # Cline uses a sliding window + "environment" context refresh.
            return self._truncate_history()
        return self.messages

    def _truncate_history(self) -> List[Dict[str, Any]]:
        # Keep last 50% of tokens? Or last N messages?
        # Let's keep last 20 messages for simplicity in this demo version
        return self.messages[-20:]

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        if not self.encoding:
            # Fallback to rough character count / 4
            return sum(len(str(m)) for m in messages) // 4

        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(self.encoding.encode(value))
                elif isinstance(value, list): # tool_calls
                    for item in value:
                        num_tokens += len(self.encoding.encode(str(item)))
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    
    def __iter__(self):
        return iter(self.messages)
