"""bridge.py -- Adapter between agent.run_stream() and Textual widgets.

The StreamBridge consumes the agent's async event generator and translates
each StreamEvent into a Textual Message that widgets can react to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncGenerator

from textual.app import App
from textual.message import Message

from koboi.events import (
    AgentDispatchEvent,
    AgentResultEvent,
    CompleteEvent,
    ErrorEvent,
    IterationEvent,
    OrchestrationCompleteEvent,
    RoutingDecisionEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)

if TYPE_CHECKING:
    from koboi.events import StreamEvent


# -- Textual Messages --------------------------------------------------------


class StreamDelta(Message):
    """A chunk of streaming text."""

    def __init__(self, content: str) -> None:
        super().__init__()
        self.content = content


class StreamToolCall(Message):
    """A tool call started."""

    def __init__(self, tool_name: str, tool_call_id: str, arguments: str) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.arguments = arguments


class StreamToolResult(Message):
    """A tool call completed."""

    def __init__(self, tool_name: str, tool_call_id: str, result: str) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.result = result


class StreamIteration(Message):
    """An iteration started."""

    def __init__(self, iteration: int, messages_count: int, tokens_estimated: int) -> None:
        super().__init__()
        self.iteration = iteration
        self.messages_count = messages_count
        self.tokens_estimated = tokens_estimated


class StreamComplete(Message):
    """The agent finished responding."""

    def __init__(self, content: str) -> None:
        super().__init__()
        self.content = content


class StreamError(Message):
    """An error occurred during streaming."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error


# -- Mode/Permission Messages ------------------------------------------------


class StreamModeChanged(Message):
    """The agent mode changed."""

    def __init__(self, old_mode: str, new_mode: str) -> None:
        super().__init__()
        self.old_mode = old_mode
        self.new_mode = new_mode


class StreamRoutingDecision(Message):
    """Router selected agents."""

    def __init__(self, agents: list[str], confidence: float, method: str, reasoning: str) -> None:
        super().__init__()
        self.agents = agents
        self.confidence = confidence
        self.method = method
        self.reasoning = reasoning


class StreamAgentDispatch(Message):
    """A sub-agent is starting."""

    def __init__(self, agent_name: str, agent_index: int, total_agents: int, mode: str) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.agent_index = agent_index
        self.total_agents = total_agents
        self.mode = mode


class StreamAgentResult(Message):
    """A sub-agent finished."""

    def __init__(self, agent_name: str, answer: str, elapsed_seconds: float, failed: bool) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.answer = answer
        self.elapsed_seconds = elapsed_seconds
        self.failed = failed


class StreamOrchestrationComplete(Message):
    """Orchestration finished with final synthesis."""

    def __init__(self, final_answer: str, elapsed_seconds: float, execution_mode: str, agent_count: int) -> None:
        super().__init__()
        self.final_answer = final_answer
        self.elapsed_seconds = elapsed_seconds
        self.execution_mode = execution_mode
        self.agent_count = agent_count


# -- Bridge ------------------------------------------------------------------


class StreamBridge:
    """Consumes agent.run_stream() and posts Textual Messages.

    Usage::

        bridge = StreamBridge(app)
        await bridge.process_stream(agent.run_stream(message))
    """

    def __init__(self, app: App) -> None:
        self._app = app
        self._is_orchestrated = False

    async def process_stream(self, stream: AsyncGenerator[StreamEvent, None]) -> None:
        """Iterate the event stream and post messages to the app."""
        self._is_orchestrated = False
        async for event in stream:
            if isinstance(event, TextDeltaEvent):
                self._app.post_message(StreamDelta(event.content))

            elif isinstance(event, ToolCallEvent):
                self._app.post_message(
                    StreamToolCall(
                        tool_name=event.tool_name,
                        tool_call_id=event.tool_call_id,
                        arguments=event.arguments,
                    )
                )

            elif isinstance(event, ToolResultEvent):
                self._app.post_message(
                    StreamToolResult(
                        tool_name=event.tool_name,
                        tool_call_id=event.tool_call_id,
                        result=event.result,
                    )
                )

            elif isinstance(event, IterationEvent):
                self._app.post_message(
                    StreamIteration(
                        iteration=event.iteration,
                        messages_count=event.messages_count,
                        tokens_estimated=event.tokens_estimated,
                    )
                )

            elif isinstance(event, CompleteEvent):
                if not self._is_orchestrated:
                    self._app.post_message(StreamComplete(content=event.content or ""))

            elif isinstance(event, ErrorEvent):
                self._app.post_message(StreamError(error=event.error))

            elif isinstance(event, RoutingDecisionEvent):
                self._is_orchestrated = True
                self._app.post_message(
                    StreamRoutingDecision(
                        agents=event.agents,
                        confidence=event.confidence,
                        method=event.method,
                        reasoning=event.reasoning,
                    )
                )

            elif isinstance(event, AgentDispatchEvent):
                self._app.post_message(
                    StreamAgentDispatch(
                        agent_name=event.agent_name,
                        agent_index=event.agent_index,
                        total_agents=event.total_agents,
                        mode=event.mode,
                    )
                )

            elif isinstance(event, AgentResultEvent):
                self._app.post_message(
                    StreamAgentResult(
                        agent_name=event.agent_name,
                        answer=event.answer,
                        elapsed_seconds=event.elapsed_seconds,
                        failed=event.failed,
                    )
                )

            elif isinstance(event, OrchestrationCompleteEvent):
                self._app.post_message(
                    StreamOrchestrationComplete(
                        final_answer=event.final_answer,
                        elapsed_seconds=event.elapsed_seconds,
                        execution_mode=event.execution_mode,
                        agent_count=len(event.agent_results),
                    )
                )
