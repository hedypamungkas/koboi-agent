from __future__ import annotations

import json
import logging
import os
from datetime import datetime


class _PlainFormatter(logging.Formatter):
    """Formatter that outputs the message as-is, no level prefix."""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


class AgentLogger:
    def __init__(self, log_dir: str = ".logs", session_id: str | None = None):
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = log_dir
        self.session_id = session_id
        self.log_path = os.path.join(log_dir, f"{session_id}.log")
        os.makedirs(log_dir, exist_ok=True)

        self._logger = logging.getLogger(f"koboi.session.{session_id}")
        self._logger.setLevel(logging.DEBUG)
        # Replace any existing file handlers (fresh session start)
        for h in list(self._logger.handlers):
            if isinstance(h, logging.FileHandler):
                h.close()
                self._logger.removeHandler(h)
        self._handler = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
        self._handler.setFormatter(_PlainFormatter())
        self._logger.addHandler(self._handler)

        self._logger.info("Koboi Agent Session: %s", session_id)
        self._logger.info("Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def close(self) -> None:
        """Close the file handler and remove it from the logger."""
        if hasattr(self, "_handler") and self._handler is not None:
            try:
                self._handler.close()
                self._logger.removeHandler(self._handler)
            except Exception:
                pass
            self._handler = None

    def __del__(self) -> None:
        self.close()

    def _append(self, text: str) -> None:
        self._logger.info(text)

    def _separator(self, title: str) -> None:
        self._append(f"\n{'=' * 80}\n[{datetime.now().strftime('%H:%M:%S')}] {title}\n{'=' * 80}\n")

    def _format_messages(self, messages: list[dict]) -> str:
        lines = [f"Messages ({len(messages)}):"]
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            preview = content[:200] + "..." if len(content) > 200 else content
            lines.append(f"  [{i}] {role}: {preview or '(no content)'}")
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    lines.append(f"       tool_call: {fn.get('name')}({fn.get('arguments')})")
            if msg.get("tool_call_id"):
                lines.append(f"       tool_call_id: {msg['tool_call_id']}")
        return "\n".join(lines)

    def log_llm_request(self, messages: list[dict], tools: list[dict] | None) -> None:
        self._separator("LLM REQUEST")
        self._append(self._format_messages(messages) + "\n")
        if tools:
            self._append(f"Tools: {[t['function']['name'] for t in tools]}\n")
        else:
            self._append("Tools: (none)\n")
        payload = {"messages": messages}
        if tools:
            payload["tools"] = tools
        self._append("\n--- Raw JSON ---\n")
        self._append(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    def log_llm_response(self, response) -> None:
        self._separator("LLM RESPONSE")
        self._append(f"Content: {response.content or '(none)'}\n")
        if response.tool_calls:
            self._append("Tool Calls:\n")
            for tc in response.tool_calls:
                self._append(f"  - {tc.name}({tc.arguments})\n")
        else:
            self._append("Tool Calls: (none)\n")

    def log_memory_snapshot(self, messages: list[dict], trigger: str) -> None:
        self._separator(f"MEMORY SNAPSHOT ({len(messages)} msgs) [{trigger}]")
        self._append(self._format_messages(messages) + "\n")

    def log_context_management(self, detail: str) -> None:
        self._separator("CONTEXT MANAGEMENT")
        self._append(f"{detail}\n")

    def log_rag_retrieval(self, query: str, results: list, method: str) -> None:
        self._separator("RAG RETRIEVAL")
        self._append(f"Query: {query}\nMethod: {method}\nResults: {len(results)}\n")
        for i, r in enumerate(results):
            preview = r.chunk.content[:150]
            self._append(f"  [{i}] score={r.score:.4f} | {preview}\n")

    def log_rag_augmentation(self, strategy: str, original: str, augmented: str, delta: int) -> None:
        self._separator("RAG AUGMENTATION")
        self._append(f"Strategy: {strategy}\nToken delta: +{delta}\n")

    def log_rag_filter(self, query: str, injected: int, mean_score: float, min_score: float, method: str) -> None:
        self._separator("RAG FILTER")
        self._append(
            f"Query: {query}\nMethod: {method}\n"
            f"min_score: {min_score} | injected: {injected} | mean_injected_score: {mean_score:.4f}\n"
        )

    def log_rag_chunking(self, doc_title: str, total: int, avg_size: float, method: str) -> None:
        self._separator("RAG CHUNKING")
        self._append(f"Doc: {doc_title} | Method: {method} | Chunks: {total} | Avg: {avg_size:.0f}\n")

    def log_rag_indexing(self, method: str, docs: int, chunks: int) -> None:
        self._separator("RAG INDEXING")
        self._append(f"Method: {method} | Docs: {docs} | Chunks: {chunks}\n")

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._append(f"[{ts}] {message}\n")

    def log_routing(self, query: str, decision) -> None:
        self._separator("ROUTING")
        self._append(f"Query: {query}\nMethod: {decision.method}\nAgents: {decision.agents}\n")

    def log_agent_dispatch(self, agent_name: str, query: str, mode: str) -> None:
        self._separator(f"AGENT DISPATCH ({agent_name})")
        self._append(f"Agent: {agent_name} | Mode: {mode}\n")

    def log_agent_result(self, result) -> None:
        self._separator(f"AGENT RESULT ({result.agent_name})")
        self._append(f"Elapsed: {result.elapsed_seconds:.2f}s | Tokens: {result.tokens_used}\n")

    def log_mcp_connect(self, command: list, info: dict) -> None:
        self._separator("MCP CONNECT")
        self._append(f"Command: {' '.join(str(c) for c in command)}\n")

    def log_mcp_discovery(self, tools: list) -> None:
        self._separator(f"MCP DISCOVERY ({len(tools)} tools)")
        for t in tools:
            name = t.name if hasattr(t, "name") else t.get("name", "?")
            self._append(f"  - {name}\n")

    def log_skill_discovery(self, skills) -> None:
        self._separator(f"SKILL DISCOVERY ({len(skills)} skills)")
        for s in skills:
            self._append(f"  - {s.name}: {s.description[:100]}\n")

    def log_skill_activation(self, skill_name: str, body_length: int) -> None:
        self._separator("SKILL ACTIVATION")
        self._append(f"Skill: {skill_name} | Body: {body_length} chars\n")

    # --- Orchestration extended logging ---

    def log_orchestration_summary(self, orch_result) -> None:
        self._separator("ORCHESTRATION SUMMARY")
        self._append(f"Query: {orch_result.query}\n")
        r = orch_result.routing
        self._append(f"Routing: {r.method} -> {r.agents}\n")
        self._append(f"Execution: {orch_result.execution_mode}\n")
        self._append(f"Agents used: {len(orch_result.agent_results)}\n")
        self._append(f"Total time: {orch_result.total_elapsed_seconds:.2f}s\n")
        for ar in orch_result.agent_results:
            self._append(
                f"  - {ar.agent_name}: {ar.elapsed_seconds:.2f}s, {ar.tokens_used} tok, {ar.revision_count} revisions\n"
            )
        self._append(f"\nFinal answer:\n{orch_result.final_answer}\n\n")

    def log_dynamic_agent_created(self, blueprint) -> None:
        self._separator("DYNAMIC AGENT CREATED")
        self._append(f"Name: {blueprint.name}\n")
        self._append(f"Domain: {blueprint.domain_label}\n")
        self._append(f"Source: {blueprint.source}\n")
        self._append(f"Chunks: {len(blueprint.chunks)}\n")
        prompt_preview = (
            blueprint.system_prompt[:300] + "..." if len(blueprint.system_prompt) > 300 else blueprint.system_prompt
        )
        self._append(f"System prompt: {prompt_preview}\n\n")

    def log_domain_analysis(self, query: str, domain_label: str, is_known: bool) -> None:
        self._separator("DOMAIN ANALYSIS")
        self._append(f"Query: {query}\n")
        self._append(f"Domain: {domain_label}\n")
        self._append(f"Known: {is_known}\n\n")

    def log_mcp_comm(self, direction: str, message: dict) -> None:
        method = message.get("method", "")
        id_ = message.get("id", "")
        label = direction.upper()
        if method:
            label += f" {method}"
        if id_:
            label += f" (id={id_})"
        self._separator(f"MCP {label}")
        self._append(json.dumps(message, indent=2, ensure_ascii=False)[:800] + "\n\n")
