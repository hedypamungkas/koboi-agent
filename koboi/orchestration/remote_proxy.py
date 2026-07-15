"""koboi/orchestration/remote_proxy.py -- a remote peer agent as an orchestration node.

When an :class:`~koboi.types.AgentDef` declares ``endpoint: <peer_name>``, the
factory builds a :class:`RemoteAgentProxy` instead of a local ``AgentCore``. The
orchestrator calls ``await node.run(query)`` on every node
(``orchestrator.py:_run_single``); the proxy POSTs the peer's ``/v1/peer/invoke``
receiver and wraps the answer in a :class:`~koboi.types.RunResult`. So a remote
koboi instance can be a first-class node in sequential/parallel/dag/conditional
graphs (distributed workflows), declared in config.

Scope: dynamic/deep_research modes rebuild agents per-query from local blueprints,
so an ``endpoint`` on those nodes is not honored (documented). On any peer failure
(unknown peer, HTTP error, timeout, malformed response) ``run()`` returns a
``RunResult`` whose ``content`` starts with ``"Error:"`` (and ``success=False``)
so the orchestration run continues -- mirroring how a failing local node becomes
``answer="Error: ..."`` and is detected by the DAG/deep_research prefix sniff.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from koboi.types import RunResult

if TYPE_CHECKING:
    from koboi.server.peers import PeerConfig, PeerRegistry

_logger = logging.getLogger(__name__)


class RemoteAgentProxy:
    """A peer koboi agent that quacks like a local orchestration node.

    The orchestrator only requires ``await node.run(query) -> RunResult`` (it reads
    ``result.content``; ``agent.memory`` is accessed once under try/except for token
    accounting, so a proxy without ``memory`` is fine).
    """

    def __init__(self, name: str, peer_name: str, peer_registry: PeerRegistry) -> None:
        self.name = name
        self.peer_name = peer_name
        self._registry = peer_registry

    async def run(self, query: str) -> RunResult:
        peer: PeerConfig | None = self._registry.get(self.peer_name)
        if peer is None:
            return RunResult(
                content=f"Error: unknown A2A peer '{self.peer_name}' for node '{self.name}'",
                success=False,
            )
        try:
            from koboi.server.peers import invoke_peer

            content = await invoke_peer(peer, query)
            return RunResult(content=content)
        except Exception as exc:  # noqa: BLE001 -- isolate peer failures so the run continues
            _logger.warning("A2A node '%s' -> peer '%s' failed: %s", self.name, self.peer_name, exc)
            return RunResult(
                content=f"Error: peer '{self.peer_name}' call failed: {exc}",
                success=False,
            )
