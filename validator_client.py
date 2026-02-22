#!/usr/bin/env python3
"""NDJSON validator client for benchmark_agent.py.

The validator protocol is line-delimited JSON (NDJSON) over stdin/stdout.
Validators must write protocol messages to stdout only; any logs belong on stderr.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


class ValidatorError(RuntimeError):
    """Raised when the external validator fails or returns invalid responses."""


@dataclass
class ValidatorRunInfo:
    provider: str
    model: str
    include_explanation: bool
    enable_cot: bool
    reasoning_effort: Optional[str]
    thinking_level: Optional[str]
    effort: Optional[str]
    max_retries: int


class ValidatorClient:
    """Persistent NDJSON subprocess client."""

    def __init__(
        self,
        command: Sequence[str],
        timeout: float = 5.0,
        debug: bool = False,
        stderr_tail_lines: int = 200,
    ) -> None:
        self._command = list(command)
        self._timeout = float(timeout)
        self._debug = bool(debug)
        self._stderr_tail_lines = int(stderr_tail_lines)

        self._proc: Optional[subprocess.Popen[str]] = None
        self._send_lock = threading.Lock()
        self._condition = threading.Condition()
        self._init_ok: Optional[Dict[str, Any]] = None
        self._results: Dict[str, Dict[str, Any]] = {}
        self._fatal_error: Optional[str] = None
        self._stderr_tail: List[str] = []

        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None

    @property
    def command(self) -> List[str]:
        return list(self._command)

    def start(self, run_info: ValidatorRunInfo) -> None:
        if self._proc is not None:
            return

        try:
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except Exception as exc:  # noqa: BLE001
            raise ValidatorError(f"Unable to start validator process: {exc}") from exc

        assert self._proc.stdout is not None
        assert self._proc.stderr is not None

        self._stdout_thread = threading.Thread(
            target=self._stdout_reader, name="validator-stdout-reader", daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_reader, name="validator-stderr-reader", daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        init_payload: Dict[str, Any] = {
            "type": "init",
            "schema_version": 1,
            "run": {
                "provider": run_info.provider,
                "model": run_info.model,
                "include_explanation": run_info.include_explanation,
                "enable_cot": run_info.enable_cot,
                "reasoning_effort": run_info.reasoning_effort,
                "thinking_level": run_info.thinking_level,
                "effort": run_info.effort,
                "max_retries": run_info.max_retries,
            },
        }
        self._send(init_payload)

        deadline = time.monotonic() + self._timeout
        with self._condition:
            while self._init_ok is None and self._fatal_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValidatorError(
                        "Timed out waiting for validator init_ok." + self._format_stderr_tail()
                    )
                self._condition.wait(timeout=remaining)

            if self._fatal_error is not None:
                raise ValidatorError(self._fatal_error + self._format_stderr_tail())

        if self._debug:
            logging.debug("Validator init_ok: %s", self._init_ok)

    def validate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_id = str(payload.get("request_id", "")).strip()
        if not request_id:
            raise ValidatorError("Validator request payload missing request_id.")

        self._ensure_alive()
        self._send(payload)

        deadline = time.monotonic() + self._timeout
        with self._condition:
            while request_id not in self._results and self._fatal_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValidatorError(
                        f"Timed out waiting for validator result for request_id={request_id!r}."
                        + self._format_stderr_tail()
                    )
                self._condition.wait(timeout=remaining)

            if self._fatal_error is not None:
                raise ValidatorError(self._fatal_error + self._format_stderr_tail())

            result = self._results.pop(request_id, None)

        if result is None:
            raise ValidatorError(
                f"Validator result for request_id={request_id!r} missing after wait."
                + self._format_stderr_tail()
            )

        if self._debug:
            logging.debug("Validator result for %s: %s", request_id, result)

        return result

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None

        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass

        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass

        try:
            proc.wait(timeout=1.0)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    def _ensure_alive(self) -> None:
        proc = self._proc
        if proc is None:
            raise ValidatorError("Validator process is not running.")
        if proc.poll() is not None:
            raise ValidatorError(
                f"Validator process exited with code {proc.returncode}." + self._format_stderr_tail()
            )
        if self._fatal_error is not None:
            raise ValidatorError(self._fatal_error + self._format_stderr_tail())

    def _send(self, payload: Dict[str, Any]) -> None:
        self._ensure_alive()
        proc = self._proc
        assert proc is not None
        assert proc.stdin is not None
        line = json.dumps(payload, ensure_ascii=False)
        if self._debug:
            logging.debug("Validator send: %s", line)
        with self._send_lock:
            try:
                proc.stdin.write(line + "\n")
                proc.stdin.flush()
            except Exception as exc:  # noqa: BLE001
                raise ValidatorError(f"Failed writing to validator stdin: {exc}") from exc

    def _stdout_reader(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return

        while True:
            try:
                line = proc.stdout.readline()
            except Exception as exc:  # noqa: BLE001
                self._set_fatal(f"Validator stdout read error: {exc}")
                return

            if line == "":
                self._set_fatal("Validator stdout closed unexpectedly.")
                return

            raw = line.strip()
            if not raw:
                continue

            try:
                message = json.loads(raw)
            except Exception as exc:  # noqa: BLE001
                self._set_fatal(f"Validator returned non-JSON on stdout: {exc}: {raw!r}")
                return

            msg_type = str(message.get("type", "")).strip()
            schema_version = message.get("schema_version")
            if schema_version != 1:
                self._set_fatal(
                    f"Unsupported validator schema_version={schema_version!r}; expected 1."
                )
                return

            with self._condition:
                if msg_type == "init_ok":
                    self._init_ok = message
                    self._condition.notify_all()
                    continue

                if msg_type == "result":
                    request_id = str(message.get("request_id", "")).strip()
                    if not request_id:
                        self._fatal_error = "Validator result missing request_id."
                        self._condition.notify_all()
                        return
                    self._results[request_id] = message
                    self._condition.notify_all()
                    continue

                self._fatal_error = f"Validator returned unknown message type={msg_type!r}."
                self._condition.notify_all()
                return

    def _stderr_reader(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return

        while True:
            try:
                line = proc.stderr.readline()
            except Exception:
                return
            if line == "":
                return
            with self._condition:
                self._stderr_tail.append(line.rstrip("\n"))
                if len(self._stderr_tail) > self._stderr_tail_lines:
                    self._stderr_tail = self._stderr_tail[-self._stderr_tail_lines :]

    def _set_fatal(self, message: str) -> None:
        with self._condition:
            self._fatal_error = message
            self._condition.notify_all()

    def _format_stderr_tail(self) -> str:
        with self._condition:
            if not self._stderr_tail:
                return ""
            tail = "\n".join(self._stderr_tail[-50:])
        return "\n\nValidator stderr (tail):\n" + tail
