"""koboi/tools/builtin/bitbucket -- Bitbucket PR tooling (create/update/get default reviewers).

CRITICAL: never implement this via a subprocess call. The Bitbucket app password/token
must be read directly in Python (config-supplied) and sent as an HTTP header on an
in-process ``httpx`` call -- never through subprocess env.

This mirrors the github.py PR tooling implementation, using Bitbucket's REST API v2.0
with Basic authentication (username=empty, password=app_password/token).
"""

from __future__ import annotations

import logging
import re

import httpx

from koboi.tools.registry import tool
from koboi.types import RiskLevel

_logger = logging.getLogger(__name__)

# Bitbucket workspace/repo slug names are URL-safe ([A-Za-z0-9._-]); validating against this
# charset rejects path/query/fragment injection.
_OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_VALID_PR_STATES = ("open", "closed", "all")


def _seg(value: str, label: str) -> str:
    v = (value or "").strip()
    if not _OWNER_REPO_RE.match(v):
        raise ValueError(f"invalid Bitbucket {label}: {value!r}")
    return v


class BitbucketClient:
    """Thin async REST client for Bitbucket pull requests (mirrors GithubClient's shape)."""

    def __init__(
        self,
        username: str,
        app_password: str,
        api_base: str = "https://api.bitbucket.org/2.0",
        timeout: int = 15,
    ) -> None:
        self._username = username
        self._app_password = app_password
        self._api_base = api_base.rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        """Return request headers with Basic authentication.

        Bitbucket uses Basic auth with username and app password. For workspace tokens,
        the username is typically empty (or any value) and the password is the token.
        """
        # Bitbucket API requires Accept header for JSON responses
        return {"Accept": "application/json"}

    def _auth(self) -> httpx.BasicAuth:
        """Return Basic authentication for Bitbucket API."""
        return httpx.BasicAuth(self._username or "", password=self._app_password)

    async def create_pr(
        self,
        workspace: str,
        repo_slug: str,
        source_branch: str,
        dest_branch: str,
        title: str,
        description: str,
        close_source_branch: bool = False,
        default_reviewers: bool = True,
    ) -> dict:
        """Create a pull request in Bitbucket.

        Args:
            workspace: Bitbucket workspace name
            repo_slug: Repository slug
            source_branch: Source branch name
            dest_branch: Destination branch name
            title: PR title
            description: PR description
            close_source_branch: Whether to close source branch after merge
            default_reviewers: Whether to add default reviewers
        """
        workspace, repo_slug = _seg(workspace, "workspace"), _seg(repo_slug, "repo_slug")
        url = f"{self._api_base}/repositories/{workspace}/{repo_slug}/pullrequests"

        payload = {
            "title": title,
            "description": description,
            "source": {"branch": {"name": source_branch}},
            "destination": {"branch": {"name": dest_branch}},
            "close_source_branch": close_source_branch,
        }

        # Add default reviewers if requested
        if default_reviewers:
            reviewers = await self.get_default_reviewers(workspace, repo_slug)
            if reviewers:
                payload["reviewers"] = reviewers

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=self._headers(), auth=self._auth())
        resp.raise_for_status()
        return resp.json()

    async def update_pr(
        self,
        workspace: str,
        repo_slug: str,
        pr_id: int,
        title: str | None = None,
        description: str | None = None,
        reviewers: list[dict] | None = None,
    ) -> dict:
        """Update a pull request in Bitbucket."""
        workspace, repo_slug = _seg(workspace, "workspace"), _seg(repo_slug, "repo_slug")
        url = f"{self._api_base}/repositories/{workspace}/{repo_slug}/pullrequests/{pr_id}"

        payload = {}
        if title is not None:
            payload["title"] = title
        if description is not None:
            payload["description"] = description
        if reviewers is not None:
            payload["reviewers"] = reviewers

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.put(url, json=payload, headers=self._headers(), auth=self._auth())
        resp.raise_for_status()
        return resp.json()

    async def get_default_reviewers(self, workspace: str, repo_slug: str) -> list[dict]:
        """Get the default reviewers for a repository.

        Returns a list of reviewer objects with 'uuid' field.
        """
        workspace, repo_slug = _seg(workspace, "workspace"), _seg(repo_slug, "repo_slug")
        url = f"{self._api_base}/repositories/{workspace}/{repo_slug}/default-reviewers"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, headers=self._headers(), auth=self._auth())
        resp.raise_for_status()
        data = resp.json()

        # Extract reviewer UUIDs from the paginated response
        reviewers = []
        for reviewer in data.get("values", []):
            if "uuid" in reviewer:
                reviewers.append({"uuid": reviewer["uuid"]})

        return reviewers

    async def get_pr(self, workspace: str, repo_slug: str, pr_id: int) -> dict:
        """Get details of a single pull request."""
        workspace, repo_slug = _seg(workspace, "workspace"), _seg(repo_slug, "repo_slug")
        url = f"{self._api_base}/repositories/{workspace}/{repo_slug}/pullrequests/{pr_id}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, headers=self._headers(), auth=self._auth())
        resp.raise_for_status()
        return resp.json()

    async def list_prs(
        self, workspace: str, repo_slug: str, state: str = "open", pagelen: int = 50
    ) -> list[dict]:
        """List pull requests for a repository."""
        workspace, repo_slug = _seg(workspace, "workspace"), _seg(repo_slug, "repo_slug")
        url = f"{self._api_base}/repositories/{workspace}/{repo_slug}/pullrequests"
        params: dict[str, str | int] = {"state": state, "pagelen": min(pagelen, 100)}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, dict) or "values" not in data:
            raise ValueError("unexpected Bitbucket response structure")
        return data.get("values", [])


def _client_or_error(_deps: dict | None) -> tuple[BitbucketClient | None, str | None]:
    client = (_deps or {}).get("bitbucket_client")
    if client is None:
        return None, "Error: Bitbucket is not configured (set bitbucket.enabled: true and bitbucket.app_password)."
    return client, None


_WORKSPACE_REPO_PARAMS = {
    "workspace": {"type": "string", "description": "Bitbucket workspace name."},
    "repo_slug": {"type": "string", "description": "Repository slug (name)."},
}


@tool(
    name="bitbucket_create_pr",
    group="bitbucket",
    description="Create a pull request on Bitbucket.",
    parameters={
        "type": "object",
        "properties": {
            **_WORKSPACE_REPO_PARAMS,
            "source_branch": {"type": "string", "description": "Source branch name, e.g. 'feature/x'."},
            "dest_branch": {"type": "string", "description": "Destination branch name, e.g. 'main'."},
            "title": {"type": "string", "description": "PR title."},
            "description": {"type": "string", "description": "PR description."},
            "close_source_branch": {
                "type": "boolean",
                "description": "Close source branch after merge. Default: false.",
            },
            "default_reviewers": {
                "type": "boolean",
                "description": "Add default reviewers to the PR. Default: true.",
            },
        },
        "required": ["workspace", "repo_slug", "source_branch", "dest_branch", "title", "description"],
    },
    risk_level=RiskLevel.DESTRUCTIVE,
    idempotent=False,  # duplicate PR creation on retry is a real, non-replayable side effect
    deps=["bitbucket_client"],
)
async def bitbucket_create_pr(
    workspace: str,
    repo_slug: str,
    source_branch: str,
    dest_branch: str,
    title: str,
    description: str = "",
    close_source_branch: bool = False,
    default_reviewers: bool = True,
    _deps: dict | None = None,
) -> str:
    client, err = _client_or_error(_deps)
    if err:
        return err
    try:
        pr = await client.create_pr(
            workspace,
            repo_slug,
            source_branch,
            dest_branch,
            title,
            description,
            close_source_branch,
            default_reviewers,
        )
    except httpx.HTTPStatusError as e:
        return f"Error: Bitbucket API returned {e.response.status_code}: {e.response.text[:300]}"
    except httpx.HTTPError as e:
        return f"Error: Bitbucket request failed: {e}"
    except ValueError as e:
        return f"Error: {e}"

    pr_id = pr.get("id", "")
    links = pr.get("links", {})
    html_url = links.get("html", {}).get("href", "")
    return f"Created PR #{pr_id}: {html_url}"


@tool(
    name="bitbucket_update_pr",
    group="bitbucket",
    description="Update a pull request's title, description, and/or reviewers on Bitbucket.",
    parameters={
        "type": "object",
        "properties": {
            **_WORKSPACE_REPO_PARAMS,
            "pr_id": {"type": "integer", "description": "Pull request ID."},
            "title": {"type": "string", "description": "New title. Omit to leave unchanged."},
            "description": {"type": "string", "description": "New description. Omit to leave unchanged."},
        },
        "required": ["workspace", "repo_slug", "pr_id"],
    },
    risk_level=RiskLevel.DESTRUCTIVE,
    idempotent=False,  # an unwanted force-update of title/description is not safely replayable
    deps=["bitbucket_client"],
)
async def bitbucket_update_pr(
    workspace: str,
    repo_slug: str,
    pr_id: int,
    title: str | None = None,
    description: str | None = None,
    _deps: dict | None = None,
) -> str:
    client, err = _client_or_error(_deps)
    if err:
        return err
    try:
        pr = await client.update_pr(workspace, repo_slug, pr_id, title=title, description=description)
    except httpx.HTTPStatusError as e:
        return f"Error: Bitbucket API returned {e.response.status_code}: {e.response.text[:300]}"
    except httpx.HTTPError as e:
        return f"Error: Bitbucket request failed: {e}"
    except ValueError as e:
        return f"Error: {e}"

    pr_id_out = pr.get("id", "")
    state = pr.get("state", "")
    links = pr.get("links", {})
    html_url = links.get("html", {}).get("href", "")
    return f"Updated PR #{pr_id_out}: state={state} {html_url}"


@tool(
    name="bitbucket_get_default_reviewers",
    group="bitbucket",
    description="Get the default reviewers for a Bitbucket repository.",
    parameters={
        "type": "object",
        "properties": _WORKSPACE_REPO_PARAMS,
        "required": ["workspace", "repo_slug"],
    },
    risk_level=RiskLevel.SAFE,
    idempotent=True,
    deps=["bitbucket_client"],
)
async def bitbucket_get_default_reviewers(
    workspace: str, repo_slug: str, _deps: dict | None = None
) -> str:
    client, err = _client_or_error(_deps)
    if err:
        return err
    try:
        reviewers = await client.get_default_reviewers(workspace, repo_slug)
    except httpx.HTTPStatusError as e:
        return f"Error: Bitbucket API returned {e.response.status_code}: {e.response.text[:300]}"
    except httpx.HTTPError as e:
        return f"Error: Bitbucket request failed: {e}"
    except ValueError as e:
        return f"Error: {e}"

    if not reviewers:
        return f"No default reviewers configured for {workspace}/{repo_slug}."
    reviewer_names = [r.get("uuid", "") for r in reviewers]
    return f"Default reviewers for {workspace}/{repo_slug}: {', '.join(reviewer_names)}"


@tool(
    name="bitbucket_list_prs",
    group="bitbucket",
    description="List pull requests on a Bitbucket repository.",
    parameters={
        "type": "object",
        "properties": {
            **_WORKSPACE_REPO_PARAMS,
            "state": {
                "type": "string",
                "enum": ["open", "closed", "all"],
                "description": "Filter by state: 'open' (default), 'closed', or 'all'.",
            },
        },
        "required": ["workspace", "repo_slug"],
    },
    risk_level=RiskLevel.SAFE,
    idempotent=True,
    deps=["bitbucket_client"],
)
async def bitbucket_list_prs(
    workspace: str, repo_slug: str, state: str = "open", _deps: dict | None = None
) -> str:
    client, err = _client_or_error(_deps)
    if err:
        return err
    if state not in _VALID_PR_STATES:
        return f"Error: state must be one of {_VALID_PR_STATES}, got {state!r}"
    try:
        prs = await client.list_prs(workspace, repo_slug, state=state)
    except httpx.HTTPStatusError as e:
        return f"Error: Bitbucket API returned {e.response.status_code}: {e.response.text[:300]}"
    except httpx.HTTPError as e:
        return f"Error: Bitbucket request failed: {e}"
    except ValueError as e:
        return f"Error: {e}"
    if not prs:
        return f"No {state} pull requests in {workspace}/{repo_slug}."
    return "\n".join(
        f"#{pr.get('id')} {pr.get('title')} ({pr.get('state')}) {pr.get('links', {}).get('html', {}).get('href', '')}"
        for pr in prs
    )


@tool(
    name="bitbucket_get_pr",
    group="bitbucket",
    description="Get details of a single pull request on Bitbucket.",
    parameters={
        "type": "object",
        "properties": {
            **_WORKSPACE_REPO_PARAMS,
            "pr_id": {"type": "integer", "description": "Pull request ID."},
        },
        "required": ["workspace", "repo_slug", "pr_id"],
    },
    risk_level=RiskLevel.SAFE,
    idempotent=True,
    deps=["bitbucket_client"],
)
async def bitbucket_get_pr(
    workspace: str, repo_slug: str, pr_id: int, _deps: dict | None = None
) -> str:
    client, err = _client_or_error(_deps)
    if err:
        return err
    try:
        pr = await client.get_pr(workspace, repo_slug, pr_id)
    except httpx.HTTPStatusError as e:
        return f"Error: Bitbucket API returned {e.response.status_code}: {e.response.text[:300]}"
    except httpx.HTTPError as e:
        return f"Error: Bitbucket request failed: {e}"
    except ValueError as e:
        return f"Error: {e}"

    author = pr.get("author", {}).get("display_name", "")
    source = pr.get("source", {}).get("branch", {}).get("name", "")
    destination = pr.get("destination", {}).get("branch", {}).get("name", "")
    links = pr.get("links", {})
    html_url = links.get("html", {}).get("href", "")
    return (
        f"#{pr.get('id')} {pr.get('title')} ({pr.get('state')})\n"
        f"author={author} source={source} destination={destination}\n"
        f"{html_url}\n\n{pr.get('description') or ''}"
    )
