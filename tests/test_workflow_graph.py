"""tests/test_workflow_graph.py -- #7: ergonomic programmatic graph builder."""

from __future__ import annotations

from koboi.orchestration.workflow_graph import WorkflowGraph
from koboi.types import AgentResponse
from tests.conftest import make_mock_response


async def test_workflow_graph_linear_build_and_invoke(mock_client):
    """Build a 3-node linear graph programmatically -> compile -> invoke."""
    g = WorkflowGraph()
    g.add_node("research", "Gather facts about the topic.")
    g.add_node("draft", "Draft from the research.")
    g.add_node("review", "Review the draft.")
    g.add_edge("research", "draft")
    g.add_edge("draft", "review")
    graph = g.compile()

    client = mock_client(
        responses=[
            AgentResponse(content="Mars is the fourth planet."),
            AgentResponse(content="Mars, the red planet, is fourth from the sun."),
            AgentResponse(content="The draft is accurate."),
            make_mock_response("synthesized-final"),
        ]
    )

    result = await graph.invoke("topic: Mars", client)

    assert result  # non-empty synthesized answer
    assert "synthesized" in result  # synthesis ran


async def test_workflow_graph_conditional_edges(mock_client):
    """Build a branching graph -> classify -> if YES then yes_branch, if NO then no_branch."""
    g = WorkflowGraph()
    g.add_node("classify", "Reply with YES or NO.")
    g.add_node("yes_branch", "Handle the YES case in one sentence.")
    g.add_node("no_branch", "Handle the NO case in one sentence.")
    g.add_conditional_edges("classify", {"YES": "yes_branch", "NO": "no_branch"})
    graph = g.compile()

    client = mock_client(
        responses=[
            AgentResponse(content="YES"),  # classify says YES
            AgentResponse(content="Yes branch handled."),  # yes_branch runs
            make_mock_response("synthesized"),
        ]
    )

    result = await graph.invoke("should I deploy?", client)

    assert result  # the YES branch executed + synthesis ran
