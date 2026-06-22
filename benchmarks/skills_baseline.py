"""benchmarks/skills_baseline.py -- Skills system baseline metrics.

Measures token overhead, routing accuracy, activation latency, and
context window impact for the skills system. Run before and after
changes to quantify improvement.

Usage:
    python benchmarks/skills_baseline.py
    python benchmarks/skills_baseline.py --output baseline_report.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from koboi.skills.registry import SkillRegistry, build_discovery_prompt, activate_skill
from koboi.tokens import estimate_tokens
from koboi.types import SkillDefinition


# ---------------------------------------------------------------------------
# Synthetic skills for scaling tests
# ---------------------------------------------------------------------------


def _make_skill(name: str, description: str, body: str = "# Skill\nDo stuff.\n") -> SkillDefinition:
    """Create a SkillDefinition with a temporary directory."""
    import tempfile

    d = Path(tempfile.mkdtemp())
    skill_dir = d / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}")
    return SkillDefinition(name=name, description=description, skill_dir=str(skill_dir))


def _generate_skills(count: int) -> list[SkillDefinition]:
    """Generate N synthetic skills with realistic descriptions."""
    templates = [
        ("code-review", "Systematic code review focusing on security, quality, and performance"),
        ("search-and-summarize", "Research a topic by searching multiple sources and summarizing findings"),
        ("incident-response", "Guide for handling security incidents and system outages"),
        ("bug-hunter", "Find and diagnose bugs in code using systematic analysis"),
        ("data-analysis", "Analyze datasets and generate insights with visualizations"),
        ("api-design", "Design RESTful APIs following best practices and OpenAPI spec"),
        ("test-writer", "Generate comprehensive unit and integration tests"),
        ("doc-generator", "Generate documentation from code comments and structure"),
        ("refactor-assistant", "Identify and apply refactoring patterns to improve code quality"),
        ("security-audit", "Perform security audit on codebase looking for vulnerabilities"),
        ("performance-optimizer", "Profile and optimize code for speed and memory usage"),
        ("migration-planner", "Plan and execute database or framework migrations"),
        ("config-manager", "Manage and validate configuration files across environments"),
        ("log-analyzer", "Parse and analyze application logs for errors and patterns"),
        ("dependency-auditor", "Check dependencies for known vulnerabilities and updates"),
        ("schema-designer", "Design database schemas with proper normalization and indexing"),
        ("ci-cd-builder", "Set up CI/CD pipelines with testing, linting, and deployment"),
        ("monitoring-setup", "Configure monitoring, alerting, and observability for services"),
        ("terraform-writer", "Generate Terraform infrastructure-as-code configurations"),
        ("docker-optimizer", "Optimize Dockerfiles for size, security, and build speed"),
        ("graphql-resolver", "Design and implement GraphQL resolvers and schemas"),
        ("event-handler", "Design event-driven architectures with proper message handling"),
        ("cache-strategist", "Design caching strategies for optimal performance"),
        ("queue-processor", "Implement message queue consumers with retry and DLQ handling"),
        ("auth-integrator", "Integrate authentication and authorization patterns"),
    ]
    skills = []
    for i in range(count):
        name, desc = templates[i % len(templates)]
        if i >= len(templates):
            name = f"{name}-{i // len(templates)}"
        skills.append(_make_skill(name, desc))
    return skills


# ---------------------------------------------------------------------------
# Metric 1: Token Overhead
# ---------------------------------------------------------------------------


def measure_token_overhead() -> dict:
    """Measure discovery prompt token cost at different skill counts."""
    results = {}
    for count in [0, 5, 10, 25, 50]:
        skills = _generate_skills(count)
        prompt = build_discovery_prompt(skills)
        chars = len(prompt)
        tokens = estimate_tokens([{"role": "system", "content": prompt}])
        results[count] = {
            "chars": chars,
            "tokens": tokens,
            "chars_per_skill": chars / count if count > 0 else 0,
            "tokens_per_skill": tokens / count if count > 0 else 0,
        }
    return results


# ---------------------------------------------------------------------------
# Metric 2: Routing Accuracy
# ---------------------------------------------------------------------------

# Test set: (query, expected_skill_name)
ROUTING_TEST_SET = [
    ("review this code for bugs", "code-review"),
    ("search for information about Python asyncio", "search-and-summarize"),
    ("there is a security incident in production", "incident-response"),
    ("find the bug in this function", "bug-hunter"),
    ("analyze this CSV data", "data-analysis"),
    ("design an API for user management", "api-design"),
    ("write tests for this module", "test-writer"),
    ("generate docs for this library", "doc-generator"),
    ("refactor this messy function", "refactor-assistant"),
    ("audit this code for security issues", "security-audit"),
    ("optimize this slow query", "performance-optimizer"),
    ("plan migration from MySQL to Postgres", "migration-planner"),
    ("check dependencies for vulnerabilities", "dependency-auditor"),
    ("set up CI/CD pipeline", "ci-cd-builder"),
    ("configure monitoring and alerts", "monitoring-setup"),
]


def measure_routing_accuracy() -> dict:
    """Measure Precision@k and Recall@k for the TF-IDF router."""
    skills = _generate_skills(25)
    registry = SkillRegistry()
    for s in skills:
        registry._skills[s.name] = s

    top_k_values = [1, 3, 5]
    results = {f"top_{k}": {"precision": 0.0, "recall": 0.0, "hits": 0, "total": 0} for k in top_k_values}

    for query, expected in ROUTING_TEST_SET:
        for k in top_k_values:
            routed = registry.route(query, top_k=k)
            routed_names = [s.name for s in routed]
            hit = expected in routed_names
            results[f"top_{k}"]["hits"] += int(hit)
            results[f"top_{k}"]["total"] += 1
            # Precision@k: 1/k if hit, 0 otherwise
            results[f"top_{k}"]["precision"] += (1.0 / k) if hit else 0.0
            # Recall@k: 1.0 if hit (single relevant doc), 0 otherwise
            results[f"top_{k}"]["recall"] += 1.0 if hit else 0.0

    for k in top_k_values:
        key = f"top_{k}"
        total = results[key]["total"]
        results[key]["precision"] = round(results[key]["precision"] / total, 3) if total else 0
        results[key]["recall"] = round(results[key]["recall"] / total, 3) if total else 0
        results[key]["hit_rate"] = round(results[key]["hits"] / total, 3) if total else 0

    return results


# ---------------------------------------------------------------------------
# Metric 3: Activation Latency
# ---------------------------------------------------------------------------


def measure_activation_latency() -> dict:
    """Measure time to activate skills (load body from disk)."""
    import tempfile

    d = Path(tempfile.mkdtemp())
    skill_dir = d / "test-skill"
    skill_dir.mkdir()
    body = "# Test Skill\n\n" + "Do stuff. " * 200  # ~2000 chars
    (skill_dir / "SKILL.md").write_text(f"---\nname: test-skill\ndescription: Test skill for benchmarking\n---\n{body}")
    skill = SkillDefinition(name="test-skill", description="Test", skill_dir=str(skill_dir))

    # Warm up
    activate_skill(skill)

    # Measure
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        skill.body = None  # Reset
        activate_skill(skill)
    elapsed = time.perf_counter() - start

    return {
        "iterations": iterations,
        "total_seconds": round(elapsed, 4),
        "avg_ms": round(elapsed / iterations * 1000, 3),
        "body_chars": len(skill.body or ""),
    }


# ---------------------------------------------------------------------------
# Metric 4: Context Window Impact
# ---------------------------------------------------------------------------


def measure_context_impact() -> dict:
    """Measure total chars injected into system prompt at different scales."""
    results = {}
    for count in [5, 10, 25, 50]:
        skills = _generate_skills(count)
        registry = SkillRegistry()
        for s in skills:
            registry._skills[s.name] = s

        prompt = registry.get_discovery_prompt()
        tokens = estimate_tokens([{"role": "system", "content": prompt}])

        results[count] = {
            "prompt_chars": len(prompt),
            "prompt_tokens": tokens,
            "pct_of_128k": round(tokens / 128000 * 100, 2),
            "pct_of_8k": round(tokens / 8000 * 100, 2),
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_baseline() -> dict:
    """Run all baseline measurements and return report."""
    print("Skills Baseline Measurement")
    print("=" * 50)

    print("\n1. Token Overhead...")
    token_overhead = measure_token_overhead()
    for count, data in token_overhead.items():
        print(f"   {count:3d} skills: {data['chars']:6d} chars, {data['tokens']:5d} tokens")

    print("\n2. Routing Accuracy...")
    routing = measure_routing_accuracy()
    for k, data in routing.items():
        print(
            f"   {k}: precision={data['precision']:.1%}, recall={data['recall']:.1%}, hit_rate={data['hit_rate']:.1%}"
        )

    print("\n3. Activation Latency...")
    latency = measure_activation_latency()
    print(f"   {latency['iterations']} iterations: avg {latency['avg_ms']:.2f}ms, body={latency['body_chars']} chars")

    print("\n4. Context Window Impact...")
    context = measure_context_impact()
    for count, data in context.items():
        print(
            f"   {count:3d} skills: {data['prompt_tokens']} tokens ({data['pct_of_128k']:.1f}% of 128K, {data['pct_of_8k']:.1f}% of 8K)"
        )

    report = {
        "token_overhead": token_overhead,
        "routing_accuracy": routing,
        "activation_latency": latency,
        "context_window_impact": context,
    }

    print("\n" + "=" * 50)
    print("Baseline complete.")
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Skills system baseline metrics")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file path")
    args = parser.parse_args()

    report = run_baseline()

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to: {output_path}")
    else:
        print("\nJSON Report:")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
