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

"""File editor for creating, editing, and reading files."""

import os
from pathlib import Path
from difflib import unified_diff


class FileEditor:
    def create_file(self, path: str, content: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        old = ""
        if p.exists():
            old = p.read_text(encoding="utf-8")
        p.write_text(content, encoding="utf-8")
        return self._diff(old, content, path)

    def edit_file(self, path: str, transform):
        p = Path(path)
        old = ""
        if p.exists():
            old = p.read_text(encoding="utf-8")
        new = transform(old)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(new, encoding="utf-8")
        return self._diff(old, new, path)

    def read_file(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8")

    def replace_in_file(self, path: str, old_str: str, new_str: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: File not found: {path}"
        
        content = p.read_text(encoding="utf-8")
        
        if old_str not in content:
            return f"Error: The string to replace was not found in {path}"
        
        if content.count(old_str) > 1:
            return f"Error: The string to replace is not unique in {path}. Found {content.count(old_str)} occurrences. Please provide more context."
            
        new_content = content.replace(old_str, new_str)
        p.write_text(new_content, encoding="utf-8")
        
        return self._diff(content, new_content, path)

    def _diff(self, old: str, new: str, path: str) -> str:
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = unified_diff(old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}")
        return "".join(diff)
