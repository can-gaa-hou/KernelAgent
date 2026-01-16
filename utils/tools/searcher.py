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

"""Code searcher for finding files and lines matching a pattern."""

import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple, Optional


class CodeSearcher:
    def __init__(self, root: Optional[str] = None):
        self.root = Path(root) if root else Path.cwd()

    def list_files(self) -> List[str]:
        result = []
        for base, _, files in os.walk(self.root):
            for f in files:
                result.append(str(Path(base) / f))
        return result

    def search(self, pattern: str, exts: Optional[Iterable[str]] = None, case_insensitive: bool = True) -> List[Tuple[str, int, str]]:
        flags = re.IGNORECASE if case_insensitive else 0
        regex = re.compile(pattern, flags)
        matches: List[Tuple[str, int, str]] = []
        for base, _, files in os.walk(self.root):
            for f in files:
                p = Path(base) / f
                if exts:
                    if p.suffix not in exts:
                        continue
                try:
                    text = p.read_text(encoding="utf-8")
                except Exception:
                    continue
                for i, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        matches.append((str(p), i, line))
        return matches

