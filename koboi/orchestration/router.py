"""Router classes for multi-agent orchestration.

Provides:
- BaseRouter: ABC for routing queries to agents
- KeywordRouter: keyword-based routing
- LLMRouter: LLM-based routing with fallback
- HybridRouter: combines keyword + LLM routing
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from koboi.types import AgentDef, RoutingDecision
from koboi.orchestration._utils import extract_json as _extract_json

if TYPE_CHECKING:
    from koboi.client import Client


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class BaseRouter(ABC):
    @abstractmethod
    async def route(self, query: str) -> RoutingDecision: ...


class KeywordRouter(BaseRouter):
    _DEFAULT_KEYWORD_MAP: dict[str, list[str]] = {
        "hr": [
            "leave",
            "allowance",
            "working hours",
            "overtime",
            "remote",
            "office",
            "uniform",
            "office",
            "training",
            "certification",
            "discipline",
            "sanction",
            "warning letter",
            "contract employee",
            "permanent employee",
            "holiday bonus",
            "probation",
            "training budget",
            "insurance",
            "social security",
            "policy",
            "guide",
            "handbook",
            "new employee",
            "benefit",
        ],
        "sales": [
            "package",
            "service",
            "price",
            "discount",
            "promo",
            "starter",
            "professional",
            "enterprise",
            "feature",
            "sla",
            "support",
            "storage",
            "ticketing",
            "knowledge base",
            "add-on",
            "premium",
            "integration",
            "migration",
            "subscription",
            "comparison",
            "subscription cost",
        ],
        "finance": [
            "payment",
            "pay",
            "invoice",
            "billing",
            "due date",
            "penalty",
            "transfer",
            "virtual account",
            "cancellation",
            "upgrade",
            "downgrade",
            "prorata",
            "refund",
            "fund",
            "terms and conditions",
            "t&c",
            "billing period",
            "notification",
        ],
    }

    def __init__(self, agent_defs: list[AgentDef] | None = None):
        if agent_defs:
            self.keyword_map = {ad.name: ad.keywords for ad in agent_defs if ad.keywords}
        else:
            self.keyword_map = self._DEFAULT_KEYWORD_MAP

    async def route(self, query: str) -> RoutingDecision:
        q = query.lower()
        matched: dict[str, int] = {}

        for agent, keywords in self.keyword_map.items():
            score = sum(1 for kw in keywords if kw in q)
            if score > 0:
                matched[agent] = score

        if not matched:
            fallback_agents = list(self.keyword_map.keys())
            return RoutingDecision(
                query=query,
                agents=fallback_agents,
                confidence=0.3,
                method="keyword",
                reasoning="No keywords matched -- broadcasting to all agents.",
            )

        max_score = max(matched.values())
        agents = sorted(matched.keys())
        confidence = min(max_score / 3.0, 1.0)

        return RoutingDecision(
            query=query,
            agents=agents,
            confidence=confidence,
            method="keyword",
            reasoning=f"Matched keywords: {matched}",
        )


class LLMRouter(BaseRouter):
    _DEFAULT_ROUTING_PROMPT = (
        "You are a router. Determine which agent should handle the question.\n\n"
        "Available agents:\n"
        "- hr: leave, allowances, working hours, remote work, employee policies, discipline, training, employee status\n"
        "- sales: service packages, pricing, discounts, product features, add-ons, package comparisons, package cost calculations\n"
        "- finance: payments, invoices, billing, penalties, upgrade/downgrade, terms and conditions, cancellations\n"
        "{dynamic_line}\n"
        "Routing examples:\n"
        '- "How much annual leave do I get?" -> {{"agents": ["hr"], "confidence": 0.95, "reasoning": "about leave"}}\n'
        '- "Calculate the cost of Enterprise package for 50 users" -> {{"agents": ["sales"], "confidence": 0.9, "reasoning": "package price calculation, not payment"}}\n'
        '- "When is the invoice due?" -> {{"agents": ["finance"], "confidence": 0.95, "reasoning": "about billing"}}\n'
        '- "Leave and package for a team of 10 people" -> {{"agents": ["hr", "sales"], "confidence": 0.85, "reasoning": "two topics: leave and package"}}\n'
        '- "Cancel subscription" -> {{"agents": ["finance"], "confidence": 0.7, "reasoning": "cancellation = finance domain"}}\n'
        '- "What is the cybersecurity incident procedure?" -> {{"agents": ["dynamic"], "confidence": 0.8, "reasoning": "IT Security domain, not a known domain", "domain_label": "IT Security"}}\n\n'
        "Question: {query}\n\n"
        'Answer ONLY JSON: {{"agents": ["hr"], "confidence": 0.9, "reasoning": "..."}}'
    )

    _DYNAMIC_LINE = "- dynamic: domains outside known agents (examples: IT, legal, facilities, events, security, etc.)"

    def __init__(
        self,
        client: Client,
        fallback: KeywordRouter | None = None,
        enable_dynamic: bool = True,
        agent_defs: list[AgentDef] | None = None,
    ):
        self.client = client
        self.fallback = fallback or KeywordRouter(agent_defs=agent_defs)
        self.enable_dynamic = enable_dynamic
        self.agent_defs = agent_defs
        self.valid_names = {ad.name for ad in agent_defs} if agent_defs else {"hr", "sales", "finance"}
        self.routing_prompt = self._build_prompt(agent_defs) if agent_defs else self._DEFAULT_ROUTING_PROMPT

    @staticmethod
    def _build_prompt(agent_defs: list[AgentDef]) -> str:
        agent_lines = "\n".join(f"- {ad.name}: {ad.description}" for ad in agent_defs)
        return (
            "You are a router. Determine which agent should handle the question.\n\n"
            "Available agents:\n"
            f"{agent_lines}\n"
            "{dynamic_line}\n"
            "Question: {query}\n\n"
            'Answer ONLY JSON: {{"agents": ["agent_name"], "confidence": 0.9, "reasoning": "..."}}'
        )

    async def route(self, query: str) -> RoutingDecision:
        dynamic_line = self._DYNAMIC_LINE if self.enable_dynamic else ""
        try:
            # .format() is inside the try because the template embeds agent
            # descriptions (f-string-interpolated at build time); a description
            # containing braces would otherwise raise and crash routing instead
            # of falling back to the keyword router.
            prompt = self.routing_prompt.format(query=query, dynamic_line=dynamic_line)
            resp = await self.client.complete(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
            )
            content = resp.content or ""
            return await self._parse_response(query, content)
        except Exception as e:
            # Never swallow silently: a brace-bearing description, a transient
            # LLM/network failure, or a parse miss all route to the keyword
            # fallback -- log so the operator knows the LLM router was bypassed.
            _logger.warning("LLM router failed (%s: %s); falling back to keyword router", type(e).__name__, e)
            return await self.fallback.route(query)

    async def _parse_response(self, query: str, content: str) -> RoutingDecision:
        data = _extract_json(content)
        if not data:
            return await self.fallback.route(query)

        try:
            agents = data.get("agents", [])
            if not agents or not isinstance(agents, list):
                return await self.fallback.route(query)

            valid = set(self.valid_names)
            if self.enable_dynamic:
                valid.add("dynamic")
            agents = [a for a in agents if a in valid]
            if not agents:
                return await self.fallback.route(query)

            domain_label = data.get("domain_label") if "dynamic" in agents else None

            return RoutingDecision(
                query=query,
                agents=agents,
                confidence=float(data.get("confidence", 0.7)),
                method="llm",
                reasoning=data.get("reasoning", ""),
                domain_label=domain_label,
            )
        except (json.JSONDecodeError, ValueError):
            return await self.fallback.route(query)


class HybridRouter(BaseRouter):
    def __init__(
        self,
        client: Client,
        confidence_threshold: float = 0.5,
        enable_dynamic: bool = True,
        agent_defs: list[AgentDef] | None = None,
    ):
        self.keyword_router = KeywordRouter(agent_defs=agent_defs)
        self.llm_router = LLMRouter(
            client=client,
            fallback=self.keyword_router,
            enable_dynamic=enable_dynamic,
            agent_defs=agent_defs,
        )
        self.confidence_threshold = confidence_threshold

    async def route(self, query: str) -> RoutingDecision:
        kw = await self.keyword_router.route(query)

        if kw.confidence >= self.confidence_threshold:
            kw.method = "hybrid(keyword)"
            # Always consult LLM to detect multi-domain or dynamic domains keyword missed
            llm = await self.llm_router.route(query)
            llm_only = [a for a in llm.agents if a not in kw.agents]
            if llm_only:
                merged = list(dict.fromkeys(kw.agents + llm_only))
                return RoutingDecision(
                    query=query,
                    agents=merged,
                    confidence=max(kw.confidence, llm.confidence),
                    method="hybrid(keyword+llm)",
                    reasoning=f"keyword: {kw.reasoning} + llm added: {llm_only}",
                    domain_label=llm.domain_label,
                )
            return kw

        llm = await self.llm_router.route(query)
        llm.method = "hybrid(llm)"
        return llm
