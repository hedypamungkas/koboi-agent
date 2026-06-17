"""koboi/harness/policy_audit.py -- JSONL audit log for policy decisions.

Writes all policy decisions to a JSONL file for compliance auditing.
Arguments are hashed (SHA-256) so sensitive data is not stored in the log.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PolicyAuditEntry:
    timestamp: float
    tool_name: str
    arguments_hash: str
    decision: str
    rule: str
    risk_level: str = ""


class PolicyAuditLog:
    def __init__(self, file_path: str = "policy_audit.jsonl", buffer_size: int = 10):
        self._path = Path(file_path)
        self._buffer: list[PolicyAuditEntry] = []
        self._buffer_size = buffer_size

    def log(
        self,
        tool_name: str,
        arguments: str,
        decision: str,
        rule: str,
        risk_level: str = "",
    ) -> None:
        entry = PolicyAuditEntry(
            timestamp=time.time(),
            tool_name=tool_name,
            arguments_hash=hashlib.sha256(arguments.encode()).hexdigest()[:16],
            decision=decision,
            rule=rule,
            risk_level=risk_level,
        )
        self._buffer.append(entry)
        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        with open(self._path, "a") as f:
            for entry in self._buffer:
                line = json.dumps({
                    "ts": entry.timestamp,
                    "tool": entry.tool_name,
                    "args_hash": entry.arguments_hash,
                    "decision": entry.decision,
                    "rule": entry.rule,
                    "risk": entry.risk_level,
                }, ensure_ascii=False)
                f.write(line + "\n")
        self._buffer.clear()

    def close(self) -> None:
        self.flush()

    @property
    def pending_count(self) -> int:
        return len(self._buffer)
