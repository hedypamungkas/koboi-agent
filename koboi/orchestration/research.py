"""koboi/orchestration/research.py -- deep-research engine primitives.

The stateful pieces used by ``Orchestrator._run_deep_research`` (``execution.mode:
deep_research``): ``ResearchBudget`` (hard caps), ``SourceStore`` (numbered citations),
``ResearchContext`` (per-run shared state, journable), and ``CoverageEvaluator`` (one LLM
judge call per depth round). Plus the research tool bundle + the synthesis/coverage prompts.

Findings flow through node *answers* (collected at the ``_run_dag_waves_with_flow`` seam), so
these primitives are self-contained -- they do not couple to the web tools. This mirrors the
GPT-Researcher shape: each research node's answer becomes a cited source ``[n]``; the final
report is synthesized from the accumulated findings with inline ``[n]`` markers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from koboi.orchestration._utils import extract_json

if TYPE_CHECKING:
    from koboi.client import Client

_logger = logging.getLogger(__name__)

# Tool bundle every planned research node receives (web I/O). Override via config.
RESEARCH_TOOLS_CONFIG: dict = {"builtin": ["web_search", "web_fetch"]}

# A2: preamble prepended to every research node's system_prompt so the LLM actually uses the
# web tools + reports sourced findings (without this, the bare step instruction under-invokes
# tools -- the repo's known tool-nudge gap).
RESEARCH_NODE_PREAMBLE = (
    "You are a research agent. To answer your assigned sub-question:\n"
    "1. Call web_search with BROAD, SIMPLE queries (2-5 words max). Do NOT use site: operators "
    "or long compound queries -- they return 0 results. If a search returns nothing, simplify.\n"
    "2. web_fetch the most relevant result URLs. If a page returns empty content (paywalled or "
    "JS-rendered), SKIP it -- do not retry the same URL. Move to the next result.\n"
    "3. After 3-5 search+fetch cycles, STOP searching and write your findings. Do not exhaust "
    "your iterations searching -- conclude as soon as you have enough evidence.\n"
    "4. Report concrete findings with the source URL for each fact. Be specific: numbers, dates, "
    "names, technical details.\n"
    "5. Never fabricate sources or facts -- if you couldn't find something, say so explicitly."
)


# W3: map research.capabilities tokens to the generate_* tools research nodes may invoke.
_CAPABILITY_TOOLS: dict[str, str] = {
    "image": "generate_image",
    "video": "generate_video",
    "music": "generate_music",
    "speech": "generate_speech",
}

_RESEARCH_MEDIA_PREAMBLE_CLAUSE = (
    "\n6. When an illustration, diagram, short video, or audio clip would clarify a finding, "
    "use the matching generate_* tool (image/video/music/speech) and cite the saved artifact "
    "path in your findings. Keep media use sparse -- only when it genuinely aids understanding."
)


def media_tools_for_capabilities(capabilities: list[str]) -> list[str]:
    """Return the ``generate_*`` tool names for the given research capability tokens.

    Unknown tokens (e.g. ``"web"``) are ignored. ``web`` is the default capability and maps to
    the web tools already in ``RESEARCH_TOOLS_CONFIG``.
    """
    return [tool for cap, tool in _CAPABILITY_TOOLS.items() if cap in capabilities]


def preamble_with_media(capabilities: list[str]) -> str:
    """Return the research node preamble, with a media clause appended when a media capability is set."""
    if any(cap in _CAPABILITY_TOOLS for cap in capabilities):
        return RESEARCH_NODE_PREAMBLE + _RESEARCH_MEDIA_PREAMBLE_CLAUSE
    return RESEARCH_NODE_PREAMBLE


# W4: automatic post-synthesis media briefing -- one structured LLM call picks per-kind
# generation prompts from the synthesized report, then the orchestrator generates them.
RESEARCH_MEDIA_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "image_prompts": {"type": "array", "items": {"type": "string"}},
        "speech_texts": {"type": "array", "items": {"type": "string"}},
        "video_prompts": {"type": "array", "items": {"type": "string"}},
        "music_prompts": {"type": "array", "items": {"type": "string"}},
    },
    "required": [],
}

_RESEARCH_MEDIA_PROMPT = """You are selecting media to accompany a research report.

For each requested kind, write concise generation prompts derived from the report's key findings:
- image_prompts: visual descriptions (a diagram, chart, or illustration of a key concept).
- speech_texts: one-sentence spoken summaries to read aloud (a voiceover).
- video_prompts: short scene descriptions for a brief clip.
- music_prompts: mood/genre descriptions for background audio.

Return ONLY JSON. Omit kinds that were not requested. Requested kinds: {kinds}.

REPORT:
{report}
"""


def build_media_selection_prompt(report: str, kinds: list[str]) -> str:
    """Build the LLM prompt asking for per-kind media generation prompts from the report."""
    return _RESEARCH_MEDIA_PROMPT.format(report=report[:4000], kinds=", ".join(kinds))


# kind -> (MediaBackend method name, RESEARCH_MEDIA_SCHEMA field, human label)
_MEDIA_KIND_DISPATCH: dict[str, tuple[str, str, str]] = {
    "image": ("generate_image", "image_prompts", "Image"),
    "speech": ("generate_speech", "speech_texts", "Audio summary"),
    "video": ("generate_video", "video_prompts", "Video"),
    "music": ("generate_music", "music_prompts", "Music"),
}


async def generate_research_media(
    client: Client,
    report: str,
    kinds: list[str],
    max_items: int,
    media_backend: object,
    logger: logging.Logger | None = None,
) -> tuple[str, list[dict]]:
    """Pick per-kind generation prompts from the report (one LLM call), then generate.

    Returns ``(media_section_text, artifacts)``. Fail-soft: any selection/generation failure yields
    fewer or no artifacts; never raises. ``media_section_text`` is empty when nothing was generated
    and is intended to be appended to the synthesized report.
    """
    if not media_backend or not kinds:
        return "", []
    try:
        from koboi.media.types import MediaRequest

        wanted = [k for k in kinds if k in _MEDIA_KIND_DISPATCH]
        if not wanted:
            return "", []
        resp = await client.complete(
            messages=[{"role": "user", "content": build_media_selection_prompt(report, wanted)}],
            response_format=RESEARCH_MEDIA_SCHEMA,
        )
        data = extract_json(resp.content or "") or {}
    except Exception as e:  # noqa: BLE001 - boundary: selection failure -> no media
        if logger:
            logger.warning("research media selection failed: %s", e)
        return "", []

    section_lines: list[str] = []
    artifacts: list[dict] = []
    for kind in kinds:
        dispatch = _MEDIA_KIND_DISPATCH.get(kind)
        if dispatch is None:
            continue
        method_name, field, label = dispatch
        generate = getattr(media_backend, method_name, None)
        if generate is None:
            continue
        for prompt in (data.get(field) or [])[:max_items]:
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            try:
                result = await generate(MediaRequest(modality=kind, prompt=prompt))
            except Exception as e:  # noqa: BLE001 - boundary: one artifact failure skips it
                if logger:
                    logger.warning("research media %s generation failed: %s", kind, e)
                continue
            if getattr(result, "status", None) != "ok":
                continue
            location = getattr(result, "local_path", None) or getattr(result, "url", None) or "(no artifact)"
            section_lines.append(f"- {label}: {prompt[:80]} -> {location}")
            cost = getattr(result, "cost_usd", None)
            artifacts.append(
                {
                    "kind": kind,
                    "prompt": prompt,
                    "local_path": getattr(result, "local_path", None),
                    "cost_usd": str(cost) if cost is not None else None,
                    "billing_unit": getattr(getattr(result, "billing_unit", None), "value", None),
                    "model": getattr(result, "model", None),
                }
            )
    if not artifacts:
        return "", []
    return "\n\n## Generated media\n" + "\n".join(section_lines), artifacts


# response_format schema for the coverage judge.
COVERAGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "overall_score": {
            "type": "number",
            "description": "Coverage score in [0,1] across all sub-questions. 1.0 = fully covered.",
        },
        "coverage": {
            "type": "object",
            "description": "Map of sub-question -> coverage score in [0,1].",
        },
        "follow_up_queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Targeted search queries to fill the remaining gaps (empty if covered).",
        },
    },
    "required": ["overall_score", "coverage", "follow_up_queries"],
}

_RESEARCH_SYNTHESIS_PROMPT = """You are synthesizing a cited research report from gathered findings.

Write a clear, well-structured report answering the original request using ONLY the findings \
below. Cite every factual claim with an inline marker like [1], [2], ... that matches a \
finding's number. Do not invent sources or use a number that does not appear in the findings. \
If the findings are insufficient to answer a part, say so explicitly rather than fabricating.

Original request:
{query}

Findings:
{findings}

Write the report now (with inline [n] citations). Do not include a sources list -- one is \
appended automatically."""

_COVERAGE_PROMPT = """You are evaluating how thoroughly gathered findings cover the research \
sub-questions.

Sub-questions:
{sub_questions}

Findings:
{findings}

Score overall coverage in [0,1] (1.0 = every sub-question is well answered), give a per \
sub-question coverage map, and list targeted follow-up search queries for any sub-question \
scored below ~0.7. If the findings fully cover the sub-questions, return overall_score=1.0 \
and an empty follow_up_queries list. IMPORTANT: If overall_score < 0.7, you MUST provide at \
least one follow_up_query describing what specific information is still missing. Never return \
an empty follow_up_queries list when coverage is insufficient."""


# ---------------------------------------------------------------------------
# ResearchBudget
# ---------------------------------------------------------------------------


@dataclass
class ResearchBudget:
    """Hard caps for a research run (modeled on DoomLoopDetector's bounded-counter pattern)."""

    max_searches: int = 15
    max_fetches: int = 20
    max_depth: int = 3
    max_tokens: int = 0  # 0 = not enforced
    used_searches: int = 0
    used_fetches: int = 0
    used_tokens: int = 0

    def remaining(self) -> bool:
        """True if the run may continue (no hard cap exceeded)."""
        if self.used_searches >= self.max_searches:
            return False
        if self.used_fetches >= self.max_fetches:
            return False
        if self.max_tokens and self.used_tokens >= self.max_tokens:
            return False
        return True

    def record_searches(self, n: int = 1) -> None:
        self.used_searches += max(0, n)

    def record_fetches(self, n: int = 1) -> None:
        self.used_fetches += max(0, n)

    def record_tokens(self, n: int) -> None:
        if n > 0:
            self.used_tokens += n


# ---------------------------------------------------------------------------
# SourceStore
# ---------------------------------------------------------------------------


@dataclass
class _Source:
    citation_id: int
    node_id: str
    text: str


@dataclass
class SourceStore:
    """Numbered-citation store over node findings (one source per node, dedup by node_id)."""

    _sources: list[_Source] = field(default_factory=list)
    _by_node: dict[str, int] = field(default_factory=dict)

    def add_findings(self, node_id: str, text: str) -> int:
        """Record/replace ``node_id``'s findings; return its stable citation id ``[n]``."""
        cleaned = (text or "").strip()
        if not cleaned:
            return 0
        existing = self._by_node.get(node_id)
        if existing is not None:
            for s in self._sources:
                if s.citation_id == existing:
                    s.text = cleaned
            return existing
        cid = len(self._sources) + 1
        self._sources.append(_Source(citation_id=cid, node_id=node_id, text=cleaned))
        self._by_node[node_id] = cid
        return cid

    def format_for_synthesis(self) -> str:
        if not self._sources:
            return "(no findings gathered)"
        return "\n\n".join(f"[{s.citation_id}] (source: {s.node_id})\n{s.text}" for s in self._sources)

    def resolve(self, n: int) -> str | None:
        for s in self._sources:
            if s.citation_id == n:
                return s.text
        return None

    def citation_ids(self) -> set[int]:
        return {s.citation_id for s in self._sources}

    def sources_list(self) -> list[dict]:
        return [{"citation_id": s.citation_id, "node_id": s.node_id} for s in self._sources]

    def sources_with_text(self) -> list[dict]:
        """Like :meth:`sources_list` but includes the source ``text`` — for RAGAS faithfulness
        (the scorer needs source TEXT as context, not just citation ids). Mirrors the
        ``to_corpus_file`` serialization shape."""
        return [{"citation_id": s.citation_id, "node_id": s.node_id, "text": s.text} for s in self._sources]

    def to_corpus_file(self, path: str) -> None:
        """Write findings as jsonl (``{citation_id, node_id, text}`` per row) for later reuse.

        A future ``_load_documents``-style loader can re-ingest this as a RAG corpus, so a
        finished research run's findings accumulate across sessions.
        """
        import json
        from pathlib import Path

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for s in self._sources:
                f.write(json.dumps({"citation_id": s.citation_id, "node_id": s.node_id, "text": s.text}) + "\n")


# ---------------------------------------------------------------------------
# ResearchContext
# ---------------------------------------------------------------------------


@dataclass
class ResearchContext:
    """Per-run shared state for a deep-research orchestration (journable via to/from_json)."""

    sub_questions: list[str] = field(default_factory=list)
    source_store: SourceStore = field(default_factory=SourceStore)
    coverage_map: dict[str, float] = field(default_factory=dict)
    budget: ResearchBudget = field(default_factory=ResearchBudget)
    depth: int = 0
    graph_run_id: str | None = None
    query: str = ""  # W5.1: original user query for resume synthesis
    final_report: str = ""  # the synthesized cited report (set after _synthesize_research)

    def add_findings(self, node_id: str, text: str) -> int:
        return self.source_store.add_findings(node_id, text)

    def to_json(self) -> str:
        return json.dumps(
            {
                "sub_questions": self.sub_questions,
                "sources": [
                    {"citation_id": s.citation_id, "node_id": s.node_id, "text": s.text}
                    for s in self.source_store._sources  # noqa: SLF001 - serialization
                ],
                "coverage_map": self.coverage_map,
                "budget": {
                    "max_searches": self.budget.max_searches,
                    "max_fetches": self.budget.max_fetches,
                    "max_depth": self.budget.max_depth,
                    "max_tokens": self.budget.max_tokens,
                    "used_searches": self.budget.used_searches,
                    "used_fetches": self.budget.used_fetches,
                    "used_tokens": self.budget.used_tokens,
                },
                "depth": self.depth,
                "graph_run_id": self.graph_run_id,
                "query": self.query,
                "final_report": self.final_report,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> ResearchContext:
        obj = json.loads(data)
        ctx = cls()
        ctx.sub_questions = list(obj.get("sub_questions") or [])
        for s in obj.get("sources") or []:
            ctx.source_store._sources.append(  # noqa: SLF001 - deserialization
                _Source(citation_id=int(s["citation_id"]), node_id=str(s["node_id"]), text=str(s["text"]))
            )
            ctx.source_store._by_node[str(s["node_id"])] = int(s["citation_id"])  # noqa: SLF001
        ctx.coverage_map = {str(k): float(v) for k, v in (obj.get("coverage_map") or {}).items()}
        b = obj.get("budget") or {}
        ctx.budget = ResearchBudget(
            max_searches=int(b.get("max_searches", 15)),
            max_fetches=int(b.get("max_fetches", 20)),
            max_depth=int(b.get("max_depth", 3)),
            max_tokens=int(b.get("max_tokens", 0)),
            used_searches=int(b.get("used_searches", 0)),
            used_fetches=int(b.get("used_fetches", 0)),
            used_tokens=int(b.get("used_tokens", 0)),
        )
        ctx.depth = int(obj.get("depth", 0))
        ctx.graph_run_id = obj.get("graph_run_id")
        ctx.query = str(obj.get("query", ""))
        ctx.final_report = str(obj.get("final_report", ""))
        return ctx


# ---------------------------------------------------------------------------
# CoverageEvaluator
# ---------------------------------------------------------------------------


class CoverageEvaluator:
    """One LLM judge call per depth round -> (overall_score, follow_up_queries, coverage_map).

    Fail-safe: any error -> score 1.0 (stop iterating). A bad judge never crashes the run.
    """

    def __init__(self, client: Client, threshold: float = 0.7) -> None:
        self._client = client
        self._threshold = threshold

    async def evaluate(self, ctx: ResearchContext) -> tuple[float, list[str], dict[str, float]]:
        # Nothing to evaluate (no sub-questions or no findings) -> considered done.
        if not ctx.sub_questions or not ctx.source_store.citation_ids():
            return 1.0, [], {}

        prompt = _COVERAGE_PROMPT.format(
            sub_questions="\n".join(f"- {q}" for q in ctx.sub_questions),
            findings=ctx.source_store.format_for_synthesis(),
        )
        try:
            resp = await self._client.complete(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                response_format=COVERAGE_SCHEMA,
            )
            data = extract_json(resp.content or "")
        except Exception as exc:  # noqa: BLE001 - judge is a boundary: any failure -> stop
            _logger.warning("CoverageEvaluator failed (%s); stopping iteration (score=1.0)", exc)
            return 1.0, [], {}

        if not isinstance(data, dict):
            return 1.0, [], {}

        try:
            score = float(data.get("overall_score", 1.0))
        except (TypeError, ValueError):
            score = 1.0
        score = max(0.0, min(1.0, score))
        follow_ups = [str(q) for q in (data.get("follow_up_queries") or [])]
        raw_map = data.get("coverage") or {}
        covmap: dict[str, float] = {}
        if isinstance(raw_map, dict):
            for k, v in raw_map.items():
                try:
                    covmap[str(k)] = max(0.0, min(1.0, float(v)))
                except (TypeError, ValueError):
                    continue
        return score, follow_ups, covmap


def build_research_synthesis_prompt(query: str, ctx: ResearchContext) -> str:
    """Build the cited-synthesis user prompt from the gathered findings."""
    return _RESEARCH_SYNTHESIS_PROMPT.format(query=query, findings=ctx.source_store.format_for_synthesis())
