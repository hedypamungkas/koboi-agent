"""koboi/tools/builtin/github -- GitHub PR tooling (create/update/list/get pull requests).

CRITICAL: never implement this via a subprocess/``gh`` CLI call. ``GITHUB_TOKEN``/``GH_TOKEN``
are hard-coded into ``koboi.harness.env.SECRET_BLOCKLIST``, so ``build_safe_env()`` strips them
from every subprocess environment unconditionally (including the sandboxed git tools' own
env). The token must be read directly in Python (config-supplied) and sent as an HTTP header
on an in-process ``httpx`` call -- never through subprocess env.

``sandbox.network``/``network_allowlist`` never sees this traffic either -- it's an in-process
``httpx`` call, the same bypass ``web_fetch``/``call_peer_agent`` already have. Trust here is
config-level (operator-set ``github.api_base``), not the subprocess network scanner.
"""

from __future__ import annotations

import logging

import httpx

from koboi.llm.auth import BearerAuth
from koboi.tools.registry import tool
from koboi.types import RiskLevel

_logger = logging.getLogger(__name__)


class GithubClient:
    """Thin async REST client for GitHub pull requests (mirrors BraveSearchProvider's shape)."""

    def __init__(self, token: str, api_base: str = "https://api.github.com", timeout: int = 15) -> None:
        self._token = token
        self._api_base = api_base.rstrip("/")
        self._timeout = timeout
        self._auth = BearerAuth(token)

    def _headers(self) -> dict[str, str]:
        return self._auth.apply({"Accept": "application/vnd.github+json"})

    async def create_pr(self, owner: str, repo: str, head: str, base: str, title: str, body: str = "") -> dict:
        url = f"{self._api_base}/repos/{owner}/{repo}/pulls"
        payload = {"title": title, "head": head, "base": base, "body": body}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    async def update_pr(
        self,
        owner: str,
        repo: str,
        number: int,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
    ) -> dict:
        url = f"{self._api_base}/repos/{owner}/{repo}/pulls/{number}"
        payload = {k: v for k, v in {"title": title, "body": body, "state": state}.items() if v is not None}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.patch(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    async def list_prs(self, owner: str, repo: str, state: str = "open", per_page: int = 30) -> list[dict]:
        url = f"{self._api_base}/repos/{owner}/{repo}/pulls"
        params = {"state": state, "per_page": min(per_page, 100)}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    async def get_pr(self, owner: str, repo: str, number: int) -> dict:
        url = f"{self._api_base}/repos/{owner}/{repo}/pulls/{number}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, headers=self._headers())
        resp.raise_for_status()
        return resp.json()


def _client_or_error(_deps: dict | None) -> tuple[GithubClient | None, str | None]:
    client = (_deps or {}).get("github_client")
    if client is None:
        return None, "Error: GitHub is not configured (set github.enabled: true and github.token)."
    return client, None


_OWNER_REPO_PARAMS = {
    "owner": {"type": "string", "description": "Repository owner (user or org)."},
    "repo": {"type": "string", "description": "Repository name."},
}


@tool(
    name="github_create_pr",
    group="github",
    description="Create a pull request on GitHub.",
    parameters={
        "type": "object",
        "properties": {
            **_OWNER_REPO_PARAMS,
            "head": {"type": "string", "description": "Branch containing the changes, e.g. 'feature/x'."},
            "base": {"type": "string", "description": "Branch to merge into, e.g. 'main'."},
            "title": {"type": "string", "description": "PR title."},
            "body": {"type": "string", "description": "PR description. Default: empty."},
        },
        "required": ["owner", "repo", "head", "base", "title"],
    },
    risk_level=RiskLevel.DESTRUCTIVE,
    idempotent=False,  # duplicate PR creation on retry is a real, non-replayable side effect
    deps=["github_client"],
)
async def github_create_pr(
    owner: str, repo: str, head: str, base: str, title: str, body: str = "", _deps: dict | None = None
) -> str:
    client, err = _client_or_error(_deps)
    if err:
        return err
    try:
        pr = await client.create_pr(owner, repo, head, base, title, body)
    except httpx.HTTPStatusError as e:
        return f"Error: GitHub API returned {e.response.status_code}: {e.response.text[:300]}"
    except httpx.HTTPError as e:
        return f"Error: GitHub request failed: {e}"
    return f"Created PR #{pr.get('number')}: {pr.get('html_url', '')}"


@tool(
    name="github_update_pr",
    group="github",
    description="Update a pull request's title, body, and/or state (open/closed) on GitHub.",
    parameters={
        "type": "object",
        "properties": {
            **_OWNER_REPO_PARAMS,
            "number": {"type": "integer", "description": "Pull request number."},
            "title": {"type": "string", "description": "New title. Omit to leave unchanged."},
            "body": {"type": "string", "description": "New description. Omit to leave unchanged."},
            "state": {"type": "string", "description": "New state: 'open' or 'closed'. Omit to leave unchanged."},
        },
        "required": ["owner", "repo", "number"],
    },
    risk_level=RiskLevel.DESTRUCTIVE,
    idempotent=False,  # an unwanted force-update of title/state/body is not safely replayable
    deps=["github_client"],
)
async def github_update_pr(
    owner: str,
    repo: str,
    number: int,
    title: str | None = None,
    body: str | None = None,
    state: str | None = None,
    _deps: dict | None = None,
) -> str:
    client, err = _client_or_error(_deps)
    if err:
        return err
    try:
        pr = await client.update_pr(owner, repo, number, title=title, body=body, state=state)
    except httpx.HTTPStatusError as e:
        return f"Error: GitHub API returned {e.response.status_code}: {e.response.text[:300]}"
    except httpx.HTTPError as e:
        return f"Error: GitHub request failed: {e}"
    return f"Updated PR #{pr.get('number')}: state={pr.get('state')} {pr.get('html_url', '')}"


@tool(
    name="github_list_prs",
    group="github",
    description="List pull requests on a GitHub repository.",
    parameters={
        "type": "object",
        "properties": {
            **_OWNER_REPO_PARAMS,
            "state": {"type": "string", "description": "Filter by state: 'open' (default), 'closed', or 'all'."},
        },
        "required": ["owner", "repo"],
    },
    risk_level=RiskLevel.SAFE,
    idempotent=True,
    deps=["github_client"],
)
async def github_list_prs(owner: str, repo: str, state: str = "open", _deps: dict | None = None) -> str:
    client, err = _client_or_error(_deps)
    if err:
        return err
    try:
        prs = await client.list_prs(owner, repo, state=state)
    except httpx.HTTPStatusError as e:
        return f"Error: GitHub API returned {e.response.status_code}: {e.response.text[:300]}"
    except httpx.HTTPError as e:
        return f"Error: GitHub request failed: {e}"
    if not prs:
        return f"No {state} pull requests in {owner}/{repo}."
    return "\n".join(
        f"#{pr.get('number')} {pr.get('title')} ({pr.get('state')}) {pr.get('html_url', '')}" for pr in prs
    )


@tool(
    name="github_get_pr",
    group="github",
    description="Get details of a single pull request on GitHub.",
    parameters={
        "type": "object",
        "properties": {
            **_OWNER_REPO_PARAMS,
            "number": {"type": "integer", "description": "Pull request number."},
        },
        "required": ["owner", "repo", "number"],
    },
    risk_level=RiskLevel.SAFE,
    idempotent=True,
    deps=["github_client"],
)
async def github_get_pr(owner: str, repo: str, number: int, _deps: dict | None = None) -> str:
    client, err = _client_or_error(_deps)
    if err:
        return err
    try:
        pr = await client.get_pr(owner, repo, number)
    except httpx.HTTPStatusError as e:
        return f"Error: GitHub API returned {e.response.status_code}: {e.response.text[:300]}"
    except httpx.HTTPError as e:
        return f"Error: GitHub request failed: {e}"
    return (
        f"#{pr.get('number')} {pr.get('title')} ({pr.get('state')})\n"
        f"head={pr.get('head', {}).get('ref')} base={pr.get('base', {}).get('ref')}\n"
        f"{pr.get('html_url', '')}\n\n{pr.get('body') or ''}"
    )
