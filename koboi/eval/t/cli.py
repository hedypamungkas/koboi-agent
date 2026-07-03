"""koboi/eval/t/cli.py -- ``koboi eval-test`` command: run ``*.eval.py`` test files."""

from __future__ import annotations

import click


@click.command("eval-test")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    "config",
    type=click.Path(),
    default=None,
    help="Agent YAML config for live runs (overrides module CONFIG).",
)
@click.option(
    "--mock/--no-mock", "mock", default=None, help="Force mock (scripted responses, no API key) or live mode."
)
@click.option("--strict", is_flag=True, help="Exit non-zero on any gate failure.")
@click.option("--threshold", type=float, default=0.6, help="Soft-score pass threshold (default: 0.6).")
@click.option("--parallel/--sequential", default=False, help="Run tests concurrently.")
@click.option("--max-concurrency", type=int, default=5, help="Max parallel tests.")
@click.option("--tags", default=None, help="Comma-separated tag filter (any-of).")
def eval_test(path, config, mock, strict, threshold, parallel, max_concurrency, tags):
    """Run eve-style ``t`` eval tests (*.eval.py files).

    Tests are ``async def test_*(t)`` functions that drive an agent and record
    assertions. With ``--strict`` the command exits non-zero if any test fails
    (gate failure or below threshold) -- suitable for CI.
    """
    from koboi.eval.t import run_tests_sync
    from koboi.eval.runner import EvalRunner

    tag_list = [tag.strip() for tag in tags.split(",")] if tags else None
    try:
        results = run_tests_sync(
            path,
            threshold=threshold,
            parallel=parallel,
            max_concurrency=max_concurrency,
            tags=tag_list,
            config=config,
            mock=mock,
        )
    except Exception as exc:  # discovery/import/config errors
        click.echo(f"eval-test error: {exc}", err=True)
        raise SystemExit(2)

    if not results:
        click.echo("No tests found.")
        raise SystemExit(2)

    click.echo(EvalRunner.format_results(results, threshold))

    failed = [result for result in results if not result.passed]
    if strict and failed:
        click.echo(f"\n{len(failed)} test(s) failed (gate failure or below threshold).", err=True)
        raise SystemExit(1)
