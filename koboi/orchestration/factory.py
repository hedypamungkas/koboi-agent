"""Agent factory and dynamic agent builder.

AgentFactory: creates pre-configured agents (hr, sales, finance, general).
DynamicAgentBuilder: builds specialist agents on-the-fly for unknown domains.
"""
from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING

from koboi.rag.chunker import ParagraphChunker
from koboi.rag.types import Chunk, Document
from koboi.orchestration._utils import extract_json as _extract_json
from koboi.types import AgentBlueprint, AgentDef

if TYPE_CHECKING:
    from koboi.client import Client
    from koboi.logger import AgentLogger
    from koboi.loop import AgentCore as Agent


CHUNKER = ParagraphChunker(max_chunk_size=1000)


# ---------------------------------------------------------------------------
# Knowledge splitting
# ---------------------------------------------------------------------------

def _split_catalog() -> tuple[str, str]:
    from koboi.rag.sample_documents import get_product_catalog
    full = get_product_catalog()
    for marker in ("4. TERMS AND CONDITIONS", "## Payment Terms"):
        idx = full.find(marker)
        if idx != -1:
            return full[:idx].strip(), full[idx:].strip()
    return full, ""


def _chunk_all() -> list[Chunk]:
    from koboi.rag.sample_documents import get_all_documents
    chunks: list[Chunk] = []
    for title, content in get_all_documents():
        doc_id = title.replace(" ", "_").lower()
        doc = Document(id=doc_id, title=title, content=content)
        chunks.extend(CHUNKER.chunk(doc))
    return chunks


def _get_hr_chunks() -> list[Chunk]:
    return [c for c in _chunk_all() if c.doc_id in ("company_policy", "employee_handbook")]


def _get_sales_chunks() -> list[Chunk]:
    sales_content, _ = _split_catalog()
    if not sales_content:
        return []
    doc = Document(id="product_catalog_sales", title="Sales Catalog", content=sales_content)
    return CHUNKER.chunk(doc)


def _get_finance_chunks() -> list[Chunk]:
    _, finance_content = _split_catalog()
    if not finance_content:
        return []
    doc = Document(id="product_catalog_finance", title="Finance Catalog", content=finance_content)
    return CHUNKER.chunk(doc)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

HR_PROMPT = (
    "You are an HR specialist agent at Acme Corp.\n"
    "You only answer questions about: leave, allowances, working hours, "
    "remote work, employee status, training, discipline, sanctions, uniforms, office locations.\n\n"
    "Rules:\n"
    "- Answer ONLY based on the provided document context.\n"
    "- Cite sources.\n"
    "- If outside the HR domain, say: 'This question is outside the HR domain.'"
)

SALES_PROMPT = (
    "You are a Sales specialist agent at Acme Corp.\n"
    "You only answer questions about: service packages (Starter, Professional, Enterprise), "
    "pricing, discounts, features, add-ons, package comparisons.\n\n"
    "Rules:\n"
    "- Answer ONLY based on the provided document context.\n"
    "- Provide specific numbers.\n"
    "- If asked to calculate, show calculation steps.\n"
    "- If outside the Sales domain, say: 'This question is outside the Sales domain.'"
)

FINANCE_PROMPT = (
    "You are a Finance specialist agent at Acme Corp.\n"
    "You answer questions about: payments, invoices, billing, "
    "late penalties, cancellations, refunds, upgrade/downgrade, terms & conditions, "
    "payment timing, invoice due dates, Net terms, deposit requirements, credit terms.\n\n"
    "Rules:\n"
    "- Answer ONLY based on the provided document context.\n"
    "- Provide specific details and numbers.\n"
    "- If the question is completely unrelated to finance/payment/billing, "
    "say: 'This question is outside the Finance domain.'"
)

GENERAL_PROMPT = (
    "You are an internal assistant at Acme Corp.\n"
    "Answer employee questions based on the provided document context.\n"
    "If information is not in the context, honestly say you don't know.\n"
    "Always cite sources."
)


# ---------------------------------------------------------------------------
# AgentFactory
# ---------------------------------------------------------------------------

class AgentFactory:
    _defaults = {
        "top_k": 3,
        "max_context_tokens": 8000,
    }

    @classmethod
    def configure(cls, **kwargs) -> None:
        cls._defaults = {**cls._defaults, **kwargs}

    @classmethod
    def create_agent(
        cls,
        agent_name: str,
        client: Client,
        logger: AgentLogger | None = None,
    ) -> Agent:
        agents = {
            "hr": (HR_PROMPT, _get_hr_chunks()),
            "sales": (SALES_PROMPT, _get_sales_chunks()),
            "finance": (FINANCE_PROMPT, _get_finance_chunks()),
            "general": (GENERAL_PROMPT, _chunk_all()),
        }
        prompt, chunks = agents.get(agent_name, agents["general"])
        return cls._build_agent(client, prompt, chunks, logger)

    @staticmethod
    def create_dynamic_agent(
        blueprint: AgentBlueprint,
        client: Client,
        logger: AgentLogger | None = None,
    ) -> Agent:
        from koboi.orchestration.factory import DynamicAgentBuilder
        builder = DynamicAgentBuilder(client=client, logger=logger)
        return builder.build_agent(blueprint)

    @classmethod
    def _build_agent(cls, client: Client, prompt: str, chunks: list[Chunk], logger: AgentLogger | None) -> Agent:
        from koboi.loop import AgentCore as Agent
        from koboi.rag.registry import retriever_registry, augmentation_registry

        ret_entry = retriever_registry.get("keyword")
        retriever = ret_entry.cls(chunks)
        aug_entry = augmentation_registry.get("in_memory")
        aug = aug_entry.cls(retriever=retriever, top_k=cls._defaults["top_k"], logger=logger)
        return Agent(
            client=client, system_prompt=prompt, augmentation=aug,
            max_context_tokens=cls._defaults["max_context_tokens"], logger=logger, verbose=False,
        )

    @classmethod
    def create_configured_agent(
        cls,
        agent_def: AgentDef,
        client: Client,
        logger: AgentLogger | None = None,
        parent_rag_config: dict | None = None,
        hook_chain: object | None = None,
    ) -> Agent:
        """Build an AgentCore from an AgentDef (config-driven)."""
        from koboi.loop import AgentCore as Agent

        augmentation = cls.build_rag_from_config(
            agent_def.rag_config, parent_rag_config, logger, client=client,
        )

        max_ctx = cls._defaults["max_context_tokens"]
        if agent_def.llm_config and "max_context_tokens" in agent_def.llm_config:
            max_ctx = agent_def.llm_config["max_context_tokens"]

        tools = cls._build_tools_from_config(agent_def.tools_config)

        return Agent(
            client=client,
            system_prompt=agent_def.system_prompt,
            augmentation=augmentation,
            max_context_tokens=max_ctx,
            logger=logger,
            verbose=False,
            tools=tools,
            hook_chain=hook_chain,
        )

    @classmethod
    def create_all_configured(
        cls,
        agent_defs: list[AgentDef],
        client: Client,
        logger: AgentLogger | None = None,
        parent_rag_config: dict | None = None,
        hook_chain: object | None = None,
    ) -> dict[str, Agent]:
        """Build all agents from config-driven AgentDef list."""
        agents = {}
        for ad in agent_defs:
            child_logger = None
            if logger:
                from koboi.logger import AgentLogger
                child_logger = AgentLogger(
                    log_dir=logger.log_dir,
                    session_id=f"{logger.session_id}_{ad.name}",
                )
            agents[ad.name] = cls.create_configured_agent(
                ad, client, child_logger, parent_rag_config,
                hook_chain=hook_chain,
            )
        return agents

    @staticmethod
    def _build_tools_from_config(tools_config: dict | None):
        """Build a ToolRegistry from agent-level tools config."""
        if not tools_config:
            return None

        from koboi.tools.registry import ToolRegistry
        from koboi.tools.builtin import register_all

        registry = ToolRegistry()
        builtin_list = tools_config.get("builtin", [])
        if builtin_list:
            register_all(registry)
            # Inject per-agent memory store so sub-agents don't share state
            from koboi.tools.builtin.memory import _MemoryStore
            memory_file = tools_config.get("memory_file", ".agent_memory.json")
            registry.set_dep("memory_store_ref", _MemoryStore(filepath=memory_file))
            registry.keep_only(builtin_list)
        return registry

    @staticmethod
    def build_rag_from_config(
        agent_rag_config: dict | None,
        parent_rag_config: dict | None,
        logger: AgentLogger | None = None,
        client: Client | None = None,
    ):
        """Build RAG augmentation from agent-level or parent-level config.

        Delegates to the RAG registry for component resolution.
        """
        rag_conf = agent_rag_config or parent_rag_config
        if not rag_conf or not rag_conf.get("enabled"):
            return None

        from koboi.rag.registry import build_rag
        return build_rag(rag_conf, client=client, logger=logger)


# ---------------------------------------------------------------------------
# DynamicAgentBuilder
# ---------------------------------------------------------------------------

KNOWN_DOMAINS = {"hr", "sales", "finance"}

DOMAIN_CLASSIFICATION_PROMPT = (
    "You are a classifier. Determine the domain of the following question.\n\n"
    "Known domains:\n"
    "- hr: leave, allowances, working hours, remote work, employee policies\n"
    "- sales: service packages, pricing, product features\n"
    "- finance: payments, invoices, billing\n\n"
    'If the question belongs to one of the above domains, answer: {{"domain": "hr", "is_known": true}}\n'
    'If the question is OUTSIDE the above domains, answer: {{"domain": "<new_domain_label>", "is_known": false}}\n\n'
    "Question: {query}\n\n"
    "Answer ONLY JSON."
)

SYSTEM_PROMPT_GENERATION_PROMPT = (
    "Create a system prompt for an AI specialist agent.\n\n"
    "Domain: {domain_label}\n"
    "Brief context: {sample_context}\n\n"
    "Format:\n"
    '1. Start with "You are a {domain_label} specialist agent at Acme Corp."\n'
    "2. Mention the topics handled.\n"
    "3. Rules:\n"
    "   - Answer ONLY based on the provided document context.\n"
    "   - Cite sources.\n"
    "   - If outside domain, state clearly.\n\n"
    "Respond in English. Only the system prompt, no additional explanation."
)

FALLBACK_DYNAMIC_PROMPT = (
    "You are a versatile AI assistant at Acme Corp.\n"
    "Domain: {domain_label}.\n\n"
    "Rules:\n"
    "- Answer ONLY based on the provided document context.\n"
    "- If information is not in the context, honestly say you don't know.\n"
    "- Always cite sources."
)


class DynamicAgentBuilder:
    def __init__(
        self,
        client: Client,
        logger: AgentLogger | None = None,
        all_chunks: list[Chunk] | None = None,
        top_k: int = 5,
        chunk_size: int = 400,
        chunk_overlap: int = 40,
        max_context_tokens: int = 8000,
    ):
        from koboi.rag.registry import retriever_registry

        self.client = client
        self.logger = logger
        self.all_chunks = all_chunks or _chunk_all()
        ret_entry = retriever_registry.get("keyword")
        self._retriever = ret_entry.cls(self.all_chunks)
        self._top_k = top_k
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_context_tokens = max_context_tokens

    async def analyze_domain(self, query: str) -> tuple[str, bool]:
        prompt = DOMAIN_CLASSIFICATION_PROMPT.format(query=query)
        try:
            resp = await self.client.complete(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
            )
            content = resp.content or ""
            data = _extract_json(content)
            if data:
                domain = data.get("domain", "general")
                is_known = bool(data.get("is_known", False))
                if self.logger:
                    self.logger.log_domain_analysis(query, domain, is_known)
                if is_known and domain.lower() in KNOWN_DOMAINS:
                    return domain.lower(), True
                return domain, False
        except (json.JSONDecodeError, KeyError, ValueError, AttributeError) as e:
            if self.logger:
                self.logger.log(f"Domain analysis LLM call failed: {e}")
        except Exception as e:
            if self.logger:
                self.logger.log(f"Unexpected error in domain analysis: {e}")
        return "general", False

    async def find_relevant_chunks(self, query: str, top_k: int | None = None) -> list[Chunk]:
        results = await self._retriever.retrieve(query, top_k=top_k or self._top_k)
        return [r.chunk for r in results if r.score > 0]

    async def generate_system_prompt(
        self, query: str, domain_label: str, sample_chunks: list[Chunk],
    ) -> str:
        sample_context = "\n".join(
            c.content[:200] for c in sample_chunks[:3]
        ) if sample_chunks else "(no context available)"

        prompt = SYSTEM_PROMPT_GENERATION_PROMPT.format(
            domain_label=domain_label, sample_context=sample_context,
        )
        try:
            resp = await self.client.complete(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
            )
            if resp.content and len(resp.content.strip()) > 50:
                return resp.content.strip()
        except Exception as e:
            if self.logger:
                self.logger.log(f"System prompt generation failed, using fallback: {e}")
        return FALLBACK_DYNAMIC_PROMPT.format(domain_label=domain_label)

    async def build_blueprint(self, query: str, domain_label: str | None = None) -> AgentBlueprint:
        if domain_label is None:
            domain_label, _ = await self.analyze_domain(query)

        chunks = await self.find_relevant_chunks(query)
        system_prompt = await self.generate_system_prompt(query, domain_label, chunks)

        safe_name = re.sub(r"[^a-z0-9_]", "_", domain_label.lower())
        agent_name = f"dynamic_{safe_name}"

        blueprint = AgentBlueprint(
            name=agent_name,
            domain_label=domain_label,
            system_prompt=system_prompt,
            chunks=chunks,
            chunker_config={"chunk_size": self._chunk_size, "overlap": self._chunk_overlap},
            retriever_top_k=self._top_k,
            source="dynamic_llm",
            created_at=time.time(),
        )

        if self.logger:
            self.logger.log_dynamic_agent_created(blueprint)

        return blueprint

    def build_agent(self, blueprint: AgentBlueprint) -> Agent:
        from koboi.loop import AgentCore as Agent
        from koboi.rag.registry import retriever_registry, augmentation_registry

        if blueprint.chunks:
            ret_entry = retriever_registry.get("keyword")
            retriever = ret_entry.cls(blueprint.chunks)
            aug_entry = augmentation_registry.get("in_memory")
            aug = aug_entry.cls(
                retriever=retriever, top_k=blueprint.retriever_top_k, logger=self.logger,
            )
        else:
            aug = None

        return Agent(
            client=self.client,
            system_prompt=blueprint.system_prompt,
            augmentation=aug,
            max_context_tokens=self._max_context_tokens,
            logger=self.logger,
            verbose=False,
        )
