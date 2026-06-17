"""Tests for koboi.logger module."""

from __future__ import annotations

import json
import os

import pytest

from koboi.logger import AgentLogger


class TestAgentLoggerConstructor:
    def test_creates_log_dir_and_file_with_correct_naming(self, tmp_path):
        """Test constructor creates log directory and file with correct naming."""
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test_session")

        # Check directory exists
        assert tmp_path.exists()

        # Check log file exists with correct naming (session_id.log)
        log_file = tmp_path / "test_session.log"
        assert log_file.exists()

    def test_session_based_file_naming(self, tmp_path):
        """Test log file naming includes session_id."""
        logger = AgentLogger(log_dir=str(tmp_path), session_id="my_session")

        log_file = tmp_path / "my_session.log"
        assert log_file.exists()

        # Filename should contain session_id
        assert "my_session" in log_file.name


class TestBasicLogging:
    def test_log_appends_timestamped_message(self, tmp_path):
        """Test log() appends timestamped message."""
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        logger.log("Test message")

        log_file = tmp_path / "test.log"
        content = log_file.read_text()

        assert "Test message" in content
        # Should have timestamp
        assert ":" in content  # Time separator

    def test_log_multiple_messages(self, tmp_path):
        """Test logging multiple messages."""
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        logger.log("Message 1")
        logger.log("Message 2")
        logger.log("Message 3")

        log_file = tmp_path / "test.log"
        content = log_file.read_text()

        assert "Message 1" in content
        assert "Message 2" in content
        assert "Message 3" in content


class TestLLMLogging:
    def test_log_llm_request_formats_messages(self, tmp_path):
        """Test log_llm_request formats messages correctly."""
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        logger.log_llm_request(messages=messages, tools=None)

        log_file = tmp_path / "test.log"
        content = log_file.read_text()

        assert "LLM REQUEST" in content
        assert "You are helpful" in content
        assert "Hello" in content

    def test_log_llm_request_with_tools(self, tmp_path):
        """Test log_llm_request formats tool definitions."""
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")

        messages = [{"role": "user", "content": "Search"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {"type": "object"},
                },
            }
        ]

        logger.log_llm_request(messages=messages, tools=tools)

        log_file = tmp_path / "test.log"
        content = log_file.read_text()

        assert "search" in content
        assert "Tools" in content


class TestMemoryLogging:
    def test_log_memory_snapshot_formats_messages(self, tmp_path):
        """Test log_memory_snapshot formats message list."""
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        logger.log_memory_snapshot(messages=messages, trigger="test_trigger")

        log_file = tmp_path / "test.log"
        content = log_file.read_text()

        assert "MEMORY SNAPSHOT" in content
        assert "test_trigger" in content
        assert "Hello" in content
        assert "Hi there" in content


class TestFormatting:
    def test_log_structured_data(self, tmp_path):
        """Test logging structured data is readable."""
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")

        data = {"key1": "value1", "key2": ["a", "b", "c"]}

        logger.log(f"Structured data: {json.dumps(data)}")

        log_file = tmp_path / "test.log"
        content = log_file.read_text()

        assert "key1" in content
        assert "value1" in content

    def test_log_unicode_content(self, tmp_path):
        """Test logging unicode content."""
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")

        logger.log("Unicode: 世界 🌍 测试")

        log_file = tmp_path / "test.log"
        content = log_file.read_text()

        assert "世界" in content
        assert "🌍" in content

    def test_log_long_messages(self, tmp_path):
        """Test logging long messages."""
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")

        long_message = "x" * 10000

        logger.log(long_message)

        log_file = tmp_path / "test.log"
        content = log_file.read_text()

        assert len(content) >= 10000


class TestSessionHandling:
    def test_multiple_sessions_same_dir(self, tmp_path):
        """Test multiple sessions in same directory."""
        logger1 = AgentLogger(log_dir=str(tmp_path), session_id="session1")
        logger2 = AgentLogger(log_dir=str(tmp_path), session_id="session2")

        logger1.log("Session 1 message")
        logger2.log("Session 2 message")

        log_file1 = tmp_path / "session1.log"
        log_file2 = tmp_path / "session2.log"

        assert log_file1.exists()
        assert log_file2.exists()

    def test_auto_session_id(self, tmp_path):
        """Test auto-generated session ID."""
        logger = AgentLogger(log_dir=str(tmp_path), session_id=None)

        assert logger.session_id is not None
        assert len(logger.session_id) > 0

        # Session ID should be in format like YYYYMMDD_HHMMSS
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 1
        assert logger.session_id in log_files[0].stem


class TestErrorHandling:
    def test_append_error_handling_oserror(self, tmp_path):
        """Test _append error handling doesn't crash on OSError."""
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")

        # Log should work normally
        logger.log("Normal message")

        log_file = tmp_path / "test.log"

        # Make file read-only to trigger OSError
        os.chmod(log_file, 0o444)

        # This should not crash, _append handles OSError
        logger.log("This might fail silently")

        # Restore permissions for cleanup
        os.chmod(log_file, 0o644)


class TestLLMResponseLogging:
    def test_log_llm_response_with_content(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        response = type("R", (), {"content": "Hello!", "tool_calls": []})()
        logger.log_llm_response(response)
        content = (tmp_path / "test.log").read_text()
        assert "LLM RESPONSE" in content
        assert "Hello!" in content

    def test_log_llm_response_with_tool_calls(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        tc = type("TC", (), {"name": "search", "arguments": '{"q": "test"}'})()
        response = type("R", (), {"content": "", "tool_calls": [tc]})()
        logger.log_llm_response(response)
        content = (tmp_path / "test.log").read_text()
        assert "search" in content

    def test_log_llm_response_no_content(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        response = type("R", (), {"content": None, "tool_calls": []})()
        logger.log_llm_response(response)
        content = (tmp_path / "test.log").read_text()
        assert "LLM RESPONSE" in content


class TestContextManagementLogging:
    def test_log_context_management(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        logger.log_context_management("Truncated 5 messages")
        content = (tmp_path / "test.log").read_text()
        assert "CONTEXT MANAGEMENT" in content
        assert "Truncated 5 messages" in content


class TestRAGLogging:
    def test_log_rag_retrieval(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        chunk = type("C", (), {"content": "Test chunk content"})()
        result = type("R", (), {"chunk": chunk, "score": 0.95})()
        logger.log_rag_retrieval("test query", [result], "keyword")
        content = (tmp_path / "test.log").read_text()
        assert "RAG RETRIEVAL" in content
        assert "test query" in content

    def test_log_rag_augmentation(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        logger.log_rag_augmentation("prepend", "original", "augmented", 100)
        content = (tmp_path / "test.log").read_text()
        assert "RAG AUGMENTATION" in content

    def test_log_rag_chunking(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        logger.log_rag_chunking("doc.md", 10, 200.0, "paragraph")
        content = (tmp_path / "test.log").read_text()
        assert "RAG CHUNKING" in content
        assert "doc.md" in content

    def test_log_rag_indexing(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        logger.log_rag_indexing("keyword", 3, 25)
        content = (tmp_path / "test.log").read_text()
        assert "RAG INDEXING" in content


class TestOrchestrationLogging:
    def test_log_routing(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        decision = type("D", (), {"method": "keyword", "agents": ["hr", "sales"]})()
        logger.log_routing("what is PTO?", decision)
        content = (tmp_path / "test.log").read_text()
        assert "ROUTING" in content

    def test_log_agent_dispatch(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        logger.log_agent_dispatch("hr-agent", "what is PTO?", "sequential")
        content = (tmp_path / "test.log").read_text()
        assert "AGENT DISPATCH" in content
        assert "hr-agent" in content

    def test_log_agent_result(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        result = type("R", (), {"agent_name": "hr", "elapsed_seconds": 1.5, "tokens_used": 100})()
        logger.log_agent_result(result)
        content = (tmp_path / "test.log").read_text()
        assert "AGENT RESULT" in content


class TestMCPLogging:
    def test_log_mcp_connect(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        logger.log_mcp_connect(["node", "server.js"], {"name": "test"})
        content = (tmp_path / "test.log").read_text()
        assert "MCP CONNECT" in content

    def test_log_mcp_discovery(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        tool = type("T", (), {"name": "echo"})()
        logger.log_mcp_discovery([tool])
        content = (tmp_path / "test.log").read_text()
        assert "MCP DISCOVERY" in content


class TestSkillLogging:
    def test_log_skill_discovery(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        skill = type("S", (), {"name": "coding", "description": "Code helper"})()
        logger.log_skill_discovery([skill])
        content = (tmp_path / "test.log").read_text()
        assert "SKILL DISCOVERY" in content

    def test_log_skill_activation(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        logger.log_skill_activation("coding", 500)
        content = (tmp_path / "test.log").read_text()
        assert "SKILL ACTIVATION" in content


class TestMessageFormatting:
    def test_format_messages_with_tool_calls(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        messages = [
            {
                "role": "assistant",
                "content": "Let me search",
                "tool_calls": [{"function": {"name": "search", "arguments": '{"q": "test"}'}}],
            }
        ]
        logger.log_memory_snapshot(messages, "test")
        content = (tmp_path / "test.log").read_text()
        assert "tool_call" in content
        assert "search" in content

    def test_format_messages_with_tool_call_id(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        messages = [{"role": "tool", "tool_call_id": "tc_123", "content": "result"}]
        logger.log_memory_snapshot(messages, "test")
        content = (tmp_path / "test.log").read_text()
        assert "tc_123" in content


class TestOrchestrationExtendedLogging:
    def test_log_orchestration_summary(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        routing = type("R", (), {"method": "keyword", "agents": ["hr"]})()
        agent_result = type(
            "AR", (), {"agent_name": "hr", "elapsed_seconds": 1.0, "tokens_used": 50, "revision_count": 0}
        )()
        orch_result = type(
            "OR",
            (),
            {
                "query": "test query",
                "routing": routing,
                "execution_mode": "sequential",
                "agent_results": [agent_result],
                "total_elapsed_seconds": 1.0,
                "final_answer": "Answer",
            },
        )()
        logger.log_orchestration_summary(orch_result)
        content = (tmp_path / "test.log").read_text()
        assert "ORCHESTRATION SUMMARY" in content

    def test_log_dynamic_agent_created(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        blueprint = type(
            "B",
            (),
            {
                "name": "finance-bot",
                "domain_label": "Finance",
                "source": "dynamic",
                "chunks": ["chunk1"],
                "system_prompt": "You are a finance assistant.",
            },
        )()
        logger.log_dynamic_agent_created(blueprint)
        content = (tmp_path / "test.log").read_text()
        assert "DYNAMIC AGENT CREATED" in content
        assert "finance" in content.lower()

    def test_log_domain_analysis(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        logger.log_domain_analysis("tax question", "Finance", is_known=True)
        content = (tmp_path / "test.log").read_text()
        assert "DOMAIN ANALYSIS" in content
        assert "Finance" in content

    def test_log_mcp_comm(self, tmp_path):
        logger = AgentLogger(log_dir=str(tmp_path), session_id="test")
        logger.log_mcp_comm("send", {"method": "initialize", "id": 1})
        content = (tmp_path / "test.log").read_text()
        assert "MCP" in content
