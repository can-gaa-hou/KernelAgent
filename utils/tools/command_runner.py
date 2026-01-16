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

"""Command runner for non-blocking subprocess execution."""

import subprocess
import threading
import time
from typing import Optional, List


class RunnerHandle:
    def __init__(self, popen: subprocess.Popen):
        self._popen = popen
        self._buffer: List[str] = []
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        for line in iter(self._popen.stdout.readline, b""):
            with self._lock:
                self._buffer.append(line.decode(errors="ignore"))
        self._popen.stdout.close()

    def is_running(self) -> bool:
        return self._popen.poll() is None

    @property
    def pid(self) -> int:
        return self._popen.pid

    def get_output(self) -> str:
        with self._lock:
            return "".join(self._buffer)

    def stop(self):
        if self.is_running():
            self._popen.terminate()
            try:
                self._popen.wait(timeout=5)
            except Exception:
                self._popen.kill()


class CommandRunner:
    def run(self, command: str, cwd: Optional[str] = None, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        return subprocess.run(command, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)

    def run_stream(self, command: str, cwd: Optional[str] = None):
        p = subprocess.Popen(command, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(p.stdout.readline, b""):
            yield line.decode(errors="ignore")
        p.stdout.close()
        p.wait()

    def run_nonblocking(self, command: str, cwd: Optional[str] = None) -> RunnerHandle:
        p = subprocess.Popen(command, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return RunnerHandle(p)

