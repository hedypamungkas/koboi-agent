"""koboi/eval/loaders/bfcl_loader.py -- BFCL dataset loader.

Loads Berkeley Function Calling Leaderboard JSONL datasets into EvalCase lists.
Source: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from koboi.types import EvalCase
from koboi.eval.loaders import DatasetLoader

_logger = logging.getLogger(__name__)


class BFCLLoader(DatasetLoader):
    """Loads BFCL JSONL datasets into EvalCase lists."""

    CATEGORIES = [
        "simple",
        "multiple",
        "parallel",
        "parallel_multiple",
        "chatting",
        "relevance",
        "rest_api",
        "sql",
        "java",
        "javascript",
    ]

    async def load(
        self,
        source: str,
        categories: list[str] | None = None,
        max_cases: int | None = None,
        **kwargs: Any,
    ) -> list[EvalCase]:
        """Load from JSONL file, JSON files, or directory.

        Supports two formats:
        - Legacy JSONL: single file with {"question", "function", "ground_truth"} per line
        - BFCL v4: directory with questions_*.json + answers_*.json (merged by id)
        """
        path = Path(source)
        cases: list[EvalCase] = []

        if path.is_dir():
            # Check for v4 format (questions_ + answers_ files)
            q_files = sorted(path.glob("questions_*.json"))
            if q_files:
                cases = await self._load_v4_dir(path, categories, max_cases)
            else:
                # Legacy JSONL format
                files = sorted(path.glob("*.jsonl"))
                for f in files:
                    cat = self._extract_category(f.name, categories)
                    if categories and cat not in categories:
                        continue
                    cases.extend(await self._load_file(f, cat))
        elif path.is_file():
            cat = self._extract_category(path.name, categories)
            cases = await self._load_file(path, cat)
        else:
            _logger.warning("BFCL source not found: %s", source)
            return []

        if max_cases and len(cases) > max_cases:
            cases = cases[:max_cases]

        return cases

    async def _load_v4_dir(
        self,
        dir_path: Path,
        categories: list[str] | None,
        max_cases: int | None,
    ) -> list[EvalCase]:
        """Load BFCL v4 format: separate questions_ and answers_ JSON files."""
        cases: list[EvalCase] = []
        q_files = sorted(dir_path.glob("questions_*.json"))

        for q_file in q_files:
            # Extract category from filename: questions_BFCL_v4_simple_python.json -> simple_python
            cat = self._extract_v4_category(q_file.name)
            if categories and cat not in categories:
                continue

            # Find matching answer file
            a_file = q_file.parent / q_file.name.replace("questions_", "answers_")
            if not a_file.exists():
                _logger.warning("No answer file for %s", q_file.name)
                continue

            # Load answers by id
            answers: dict[str, list] = {}
            with open(a_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    answers[entry["id"]] = entry.get("ground_truth", [])

            # Load questions and merge with answers
            with open(q_file) as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    entry_id = entry.get("id", f"{cat}_{i}")
                    gt = answers.get(entry_id, [])
                    case = self._parse_v4_entry(entry, gt, entry_id, cat)
                    if case:
                        cases.append(case)

            _logger.info("Loaded %d cases from %s", len(answers), q_file.name)

        return cases

    def _parse_v4_entry(self, entry: dict, ground_truth: list, name: str, category: str) -> EvalCase | None:
        """Parse a BFCL v4 entry (separate question + answer format)."""
        # Parse question - v4 format: [[{"role":"user","content":"..."}]]
        question = entry.get("question", [])
        if isinstance(question, list):
            if question and isinstance(question[0], list):
                # Nested: [[msg1, msg2]] -> join content
                user_message = " ".join(m.get("content", "") for m in question[0] if isinstance(m, dict))
            else:
                user_message = " ".join(str(q) for q in question)
        else:
            user_message = str(question)

        # Parse function definitions
        functions = entry.get("function", [])
        if isinstance(functions, dict):
            functions = [functions]
        tool_definitions = [self._parse_function_def(fn) for fn in functions]

        # Parse ground truth - v4 format: [{"func_name": {"arg1": [val1, val2], ...}}]
        expected_tool_calls = self._parse_v4_ground_truth(ground_truth)

        return EvalCase(
            name=name,
            user_message=user_message,
            tool_definitions=tool_definitions,
            expected_tool_calls=expected_tool_calls,
            tags=["bfcl", category],
            metadata={"category": category, "source": "bfcl_v4"},
        )

    @staticmethod
    def _parse_v4_ground_truth(gt: list) -> list[dict]:
        """Parse BFCL v4 ground truth format.

        Format: [{"func_name": {"arg1": [acceptable_val1, acceptable_val2], ...}}]
        Each top-level dict = one acceptable function call.
        """
        result: list[dict] = []
        for item in gt:
            if not isinstance(item, dict):
                continue
            for func_name, args in item.items():
                # Take first acceptable value for each arg
                parsed_args = {}
                if isinstance(args, dict):
                    for k, v in args.items():
                        if isinstance(v, list) and v:
                            parsed_args[k] = v[0]
                        else:
                            parsed_args[k] = v
                result.append({"name": BFCLLoader._sanitize_name(func_name), "arguments": parsed_args})
        return result

    @staticmethod
    def _extract_v4_category(filename: str) -> str:
        """Extract category from BFCL v4 filename."""
        name = Path(filename).stem.lower()
        # questions_BFCL_v4_simple_python -> simple_python
        for prefix in ("questions_bfcl_v4_", "answers_bfcl_v4_", "bfcl_v4_"):
            if prefix in name:
                return name.split(prefix, 1)[1]
        return "unknown"

    def framework_name(self) -> str:
        return "bfcl"

    async def _load_file(self, path: Path, category: str) -> list[EvalCase]:
        cases: list[EvalCase] = []
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    case = self._parse_entry(entry, f"{category}_{i}", category)
                    if case:
                        cases.append(case)
                except json.JSONDecodeError:
                    _logger.warning("Invalid JSON at %s:%d", path, i)
        return cases

    def _parse_entry(self, entry: dict, name: str, category: str) -> EvalCase | None:
        # Parse question
        question = entry.get("question", [])
        if isinstance(question, list):
            user_message = " ".join(str(q) for q in question)
        else:
            user_message = str(question)

        # Parse function definitions
        functions = entry.get("function", [])
        if isinstance(functions, dict):
            functions = [functions]
        tool_definitions = [self._parse_function_def(fn) for fn in functions]

        # Parse ground truth
        ground_truth = entry.get("ground_truth", [])
        expected_tool_calls = self._parse_ground_truth(ground_truth)

        return EvalCase(
            name=name,
            user_message=user_message,
            tool_definitions=tool_definitions,
            expected_tool_calls=expected_tool_calls,
            tags=["bfcl", category],
            metadata={"category": category, "source": "bfcl"},
        )

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize tool name: replace dots with underscores for API compatibility."""
        return name.replace(".", "_")

    @staticmethod
    def _parse_function_def(fn_def: dict) -> dict:
        """Convert BFCL function definition to OpenAI tool format."""
        # BFCL format can vary; normalize to OpenAI function calling format
        if "type" in fn_def and fn_def["type"] == "function":
            tool = fn_def
        else:
            # Convert from BFCL's direct function format
            name = fn_def.get("name", "")
            description = fn_def.get("description", "")
            parameters = fn_def.get("parameters", {})
            tool = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }

        # Sanitize tool name (dots break some API proxies)
        func = tool.get("function", {})
        if "name" in func:
            func["name"] = BFCLLoader._sanitize_name(func["name"])

        # Normalize parameters.type: "dict" -> "object" (OpenAI requirement)
        params = func.get("parameters", {})
        if params.get("type") == "dict":
            params["type"] = "object"

        return tool

    @staticmethod
    def _parse_ground_truth(gt: list | dict) -> list[dict]:
        """Parse expected function calls from ground_truth field.

        Ground truth formats:
        - Simple: [{"name": "func", "arguments": {"key": "val"}}]
        - Multiple: [[{"name": "f1", ...}, {"name": "f2", ...}]]  (parallel)
        - String: ["func(arg1=val1, arg2=val2)"]  (Python-style)
        """
        if not gt:
            return []

        if isinstance(gt, dict):
            gt = [gt]

        result: list[dict] = []

        # Handle nested list (parallel calls)
        if gt and isinstance(gt[0], list):
            gt = gt[0]

        for item in gt:
            if isinstance(item, dict):
                result.append(
                    {
                        "name": BFCLLoader._sanitize_name(item.get("name", "")),
                        "arguments": item.get("arguments", {}),
                    }
                )
            elif isinstance(item, str):
                # Parse Python-style function call
                parsed = _parse_python_call(item)
                if parsed:
                    parsed["name"] = BFCLLoader._sanitize_name(parsed["name"])
                    result.append(parsed)

        return result

    @staticmethod
    def _extract_category(filename: str, allowed: list[str] | None) -> str:
        """Extract category from BFCL filename."""
        name = Path(filename).stem.lower()
        for cat in BFCLLoader.CATEGORIES:
            if cat in name:
                return cat
        return "unknown"


def _parse_python_call(call_str: str) -> dict | None:
    """Parse a Python-style function call string like 'func(arg1=val1, arg2=val2)'."""
    import re

    match = re.match(r"(\w+)\((.*)\)$", call_str.strip())
    if not match:
        return None

    func_name = match.group(1)
    args_str = match.group(2).strip()

    if not args_str:
        return {"name": func_name, "arguments": {}}

    arguments: dict[str, Any] = {}
    # Simple arg parsing - handles key=value pairs
    for arg_match in re.finditer(r"(\w+)\s*=\s*(.+?)(?:,\s*(?=\w+=)|$)", args_str):
        key = arg_match.group(1)
        val_str = arg_match.group(2).strip().strip("\"'")
        # Try to parse as JSON value
        try:
            val = json.loads(val_str)
        except (json.JSONDecodeError, ValueError):
            val = val_str
        arguments[key] = val

    return {"name": func_name, "arguments": arguments}
