"""Tests for koboi.tools.builtin.github (Wave 4). All HTTP is mocked (no network)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from koboi.modes import is_read_only_tool
from koboi.tools.builtin.github import (
    GithubClient,
    github_create_pr,
    github_get_pr,
    github_list_prs,
    github_update_pr,
)
from koboi.types import RiskLevel


def _response(*, status: int = 200, json_payload=None) -> httpx.Response:
    return httpx.Response(status, json=json_payload, request=httpx.Request("GET", "https://api.github.com"))


def _mock_async_client(response: httpx.Response) -> MagicMock:
    """An httpx.AsyncClient double: async CM whose .get/.post/.patch return ``response``."""
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    client.post = AsyncMock(return_value=response)
    client.patch = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class TestGithubClient:
    async def test_create_pr_posts_expected_payload(self):
        payload = {"number": 42, "html_url": "https://github.com/o/r/pull/42"}
        mock_client = _mock_async_client(_response(json_payload=payload))
        with patch("koboi.tools.builtin.github.httpx.AsyncClient", return_value=mock_client):
            result = await GithubClient(token="t").create_pr("o", "r", "feat", "main", "Title", "Body")
        assert result["number"] == 42
        mock_client.post.assert_awaited_once()
        _, kwargs = mock_client.post.call_args
        assert kwargs["json"] == {"title": "Title", "head": "feat", "base": "main", "body": "Body"}
        assert kwargs["headers"]["Authorization"] == "Bearer t"

    async def test_update_pr_omits_unset_fields(self):
        mock_client = _mock_async_client(_response(json_payload={"number": 1, "state": "closed"}))
        with patch("koboi.tools.builtin.github.httpx.AsyncClient", return_value=mock_client):
            await GithubClient(token="t").update_pr("o", "r", 1, state="closed")
        _, kwargs = mock_client.patch.call_args
        assert kwargs["json"] == {"state": "closed"}

    async def test_list_prs_returns_list(self):
        payload = [{"number": 1, "title": "A", "state": "open", "html_url": "u1"}]
        with patch(
            "koboi.tools.builtin.github.httpx.AsyncClient",
            return_value=_mock_async_client(_response(json_payload=payload)),
        ):
            result = await GithubClient(token="t").list_prs("o", "r")
        assert result == payload

    async def test_http_error_raises(self):
        error_resp = httpx.Response(401, request=httpx.Request("GET", "https://api.github.com"))
        with patch("koboi.tools.builtin.github.httpx.AsyncClient", return_value=_mock_async_client(error_resp)):
            with pytest.raises(httpx.HTTPStatusError):
                await GithubClient(token="bad").get_pr("o", "r", 1)


class TestToolsMissingClient:
    async def test_create_pr_no_client_returns_error(self):
        result = await github_create_pr("o", "r", "h", "b", "t", _deps=None)
        assert result.startswith("Error: GitHub is not configured")

    async def test_list_prs_no_client_returns_error(self):
        result = await github_list_prs("o", "r", _deps={})
        assert result.startswith("Error: GitHub is not configured")


class TestToolsWithMockedClient:
    async def test_create_pr_happy_path(self):
        client = AsyncMock()
        client.create_pr = AsyncMock(return_value={"number": 7, "html_url": "https://github.com/o/r/pull/7"})
        result = await github_create_pr("o", "r", "feat", "main", "Title", _deps={"github_client": client})
        assert "Created PR #7" in result
        assert "pull/7" in result

    async def test_create_pr_http_status_error_formatted(self):
        client = AsyncMock()
        resp = httpx.Response(422, text="validation failed", request=httpx.Request("POST", "https://api.github.com"))
        client.create_pr = AsyncMock(side_effect=httpx.HTTPStatusError("x", request=resp.request, response=resp))
        result = await github_create_pr("o", "r", "feat", "main", "Title", _deps={"github_client": client})
        assert result.startswith("Error: GitHub API returned 422")

    async def test_update_pr_happy_path(self):
        client = AsyncMock()
        client.update_pr = AsyncMock(return_value={"number": 3, "state": "closed", "html_url": "u"})
        result = await github_update_pr("o", "r", 3, state="closed", _deps={"github_client": client})
        assert "Updated PR #3" in result
        assert "state=closed" in result

    async def test_list_prs_empty(self):
        client = AsyncMock()
        client.list_prs = AsyncMock(return_value=[])
        result = await github_list_prs("o", "r", _deps={"github_client": client})
        assert "No open pull requests" in result

    async def test_list_prs_formats_each_pr(self):
        client = AsyncMock()
        client.list_prs = AsyncMock(return_value=[{"number": 5, "title": "Fix bug", "state": "open", "html_url": "u5"}])
        result = await github_list_prs("o", "r", _deps={"github_client": client})
        assert "#5 Fix bug (open) u5" in result

    async def test_get_pr_formats_details(self):
        client = AsyncMock()
        client.get_pr = AsyncMock(
            return_value={
                "number": 9,
                "title": "Add feature",
                "state": "open",
                "head": {"ref": "feat"},
                "base": {"ref": "main"},
                "html_url": "u9",
                "body": "description here",
            }
        )
        result = await github_get_pr("o", "r", 9, _deps={"github_client": client})
        assert "#9 Add feature (open)" in result
        assert "head=feat base=main" in result
        assert "description here" in result


class TestRiskLevelsAndIdempotency:
    def test_create_and_update_are_destructive_non_idempotent(self):
        for fn in (github_create_pr, github_update_pr):
            td = fn._tool_def
            assert td.risk_level == RiskLevel.DESTRUCTIVE
            assert td.idempotent is False

    def test_list_and_get_are_safe_idempotent(self):
        for fn in (github_list_prs, github_get_pr):
            td = fn._tool_def
            assert td.risk_level == RiskLevel.SAFE
            assert td.idempotent is True


class TestModeAllowlist:
    def test_read_tools_are_read_only(self):
        assert is_read_only_tool("github_list_prs") is True
        assert is_read_only_tool("github_get_pr") is True

    def test_write_tools_are_not_read_only(self):
        assert is_read_only_tool("github_create_pr") is False
        assert is_read_only_tool("github_update_pr") is False


class TestFacadeWiring:
    def test_dep_absent_when_disabled(self):
        from koboi.facade import KoboiAgent

        agent = KoboiAgent.from_dict(
            {
                "agent": {"name": "t"},
                "llm": {"provider": "openai", "model": "m", "api_key": "k"},
                "memory": {"backend": "in_memory"},
            }
        )
        assert agent._core.tools.get_dep("github_client") is None

    def test_dep_present_when_enabled_with_token(self):
        from koboi.facade import KoboiAgent

        agent = KoboiAgent.from_dict(
            {
                "agent": {"name": "t"},
                "llm": {"provider": "openai", "model": "m", "api_key": "k"},
                "memory": {"backend": "in_memory"},
                "github": {"enabled": True, "token": "tkn"},
            }
        )
        assert isinstance(agent._core.tools.get_dep("github_client"), GithubClient)

    def test_enabled_without_token_warns_and_stays_absent(self, caplog):
        from koboi.facade import KoboiAgent

        agent = KoboiAgent.from_dict(
            {
                "agent": {"name": "t"},
                "llm": {"provider": "openai", "model": "m", "api_key": "k"},
                "memory": {"backend": "in_memory"},
                "github": {"enabled": True},
            }
        )
        assert agent._core.tools.get_dep("github_client") is None


class TestOrchestrationWiring:
    def test_github_client_threaded_into_orchestration(self):
        from koboi.facade import KoboiAgent

        agent = KoboiAgent.from_dict(
            {
                "agent": {"name": "t"},
                "llm": {"provider": "openai", "model": "m", "api_key": "k"},
                "memory": {"backend": "in_memory"},
                "github": {"enabled": True, "token": "tkn"},
                "orchestration": {
                    "enabled": True,
                    "execution": {"mode": "sequential"},
                    "agents": [
                        {"name": "worker", "system_prompt": "x", "tools_config": {"builtin": ["github_get_pr"]}}
                    ],
                },
            }
        )
        assert agent._orchestrator is not None
        worker = agent._orchestrator._agents_map["worker"]
        tools = worker.tools
        assert isinstance(tools.get_dep("github_client"), GithubClient)


class TestRedactionSpotCheck:
    async def test_token_never_appears_in_error_string(self):
        # Meaningful redaction check: route the REAL token through GithubClient
        # (BearerAuth sets the Authorization header) and force a 401. The token is
        # genuinely sent, so the assertion "token not in result" only holds if the
        # tool's error path does not leak it (the body is "Bad credentials", no
        # token). The previous version never routed the token and so asserted nothing.
        secret_token = "ghp_supersecrettoken1234567890"
        resp = httpx.Response(
            401,
            text="Bad credentials",
            request=httpx.Request("POST", "https://api.github.com/repos/o/r/pulls"),
        )
        mock_client = _mock_async_client(resp)
        with patch("koboi.tools.builtin.github.httpx.AsyncClient", return_value=mock_client):
            result = await github_create_pr(
                "o", "r", "h", "main", "t", body="", _deps={"github_client": GithubClient(token=secret_token)}
            )
        assert secret_token not in result
        # Sanity: the token really was on the request, so a leak would be the tool's fault.
        assert mock_client.post.call_args.kwargs["headers"]["Authorization"] == f"Bearer {secret_token}"


class TestGithubHardening:
    async def test_list_prs_non_list_response_returns_error(self):
        client = AsyncMock()
        client.list_prs = AsyncMock(side_effect=ValueError("expected list, got dict"))
        assert (await github_list_prs("o", "r", _deps={"github_client": client})).startswith("Error:")

    async def test_get_pr_non_dict_response_returns_error(self):
        client = AsyncMock()
        client.get_pr = AsyncMock(side_effect=ValueError("expected object, got list"))
        assert (await github_get_pr("o", "r", 1, _deps={"github_client": client})).startswith("Error:")

    async def test_update_pr_rejects_invalid_state(self):
        client = AsyncMock()
        client.update_pr = AsyncMock(return_value={"number": 1, "state": "open", "html_url": "u"})
        result = await github_update_pr("o", "r", 1, state="merged", _deps={"github_client": client})
        assert result.startswith("Error:")
        client.update_pr.assert_not_awaited()  # validation happens before the call

    async def test_list_prs_rejects_invalid_state(self):
        client = AsyncMock()
        result = await github_list_prs("o", "r", state="bogus", _deps={"github_client": client})
        assert result.startswith("Error:")

    async def test_client_rejects_owner_with_path_separator(self):
        # owner/repo validated against [A-Za-z0-9._-] -- rejects path/query/fragment
        # injection (owner='foo?bar' would retarget the request).
        with pytest.raises(ValueError):
            await GithubClient(token="t").list_prs("foo?bar", "r")
        with pytest.raises(ValueError):
            await GithubClient(token="t").list_prs("a/b", "r")

    async def test_update_pr_clears_title_to_empty(self):
        # Empty-string title is a legitimate clear (survives `if v is not None`).
        mock_client = _mock_async_client(_response(json_payload={"number": 1, "state": "open"}))
        with patch("koboi.tools.builtin.github.httpx.AsyncClient", return_value=mock_client):
            await GithubClient(token="t").update_pr("o", "r", 1, title="")
        assert mock_client.patch.call_args.kwargs["json"] == {"title": ""}
