"""Tests for koboi.tools.builtin.bitbucket (contribution #6). All HTTP is mocked (no network)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from koboi.tools.builtin.bitbucket import (
    BitbucketClient,
    bitbucket_create_pr,
    bitbucket_get_default_reviewers,
    bitbucket_get_pr,
    bitbucket_list_prs,
    bitbucket_update_pr,
)


def _response(*, status: int = 200, json_payload=None) -> httpx.Response:
    return httpx.Response(status, json=json_payload, request=httpx.Request("GET", "https://api.bitbucket.org/2.0"))


def _mock_async_client(response: httpx.Response) -> MagicMock:
    """An httpx.AsyncClient double: async CM whose .get/.post/.patch return ``response``."""
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    client.post = AsyncMock(return_value=response)
    client.put = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class TestBitbucketClient:
    async def test_create_pr_posts_expected_payload(self):
        payload = {"id": 42, "links": {"html": {"href": "https://bitbucket.org/workspace/repo/pullrequests/42"}}}
        mock_client = _mock_async_client(_response(json_payload=payload))
        with patch("koboi.tools.builtin.bitbucket.httpx.AsyncClient", return_value=mock_client):
            result = await BitbucketClient(username="user", app_password="token").create_pr(
                workspace="ws",
                repo_slug="repo",
                source_branch="feat",
                dest_branch="main",
                title="Title",
                description="Body",
            )
        assert result["id"] == 42
        mock_client.post.assert_awaited_once()
        _, kwargs = mock_client.post.call_args
        assert kwargs["json"]["title"] == "Title"
        assert kwargs["json"]["source"]["branch"]["name"] == "feat"
        assert kwargs["json"]["destination"]["branch"]["name"] == "main"

    async def test_create_pr_includes_default_reviewers_when_enabled(self):
        payload = {"id": 1, "links": {"html": {"href": "https://bitbucket.org/ws/repo/pullrequests/1"}}}
        mock_client = _mock_async_client(_response(json_payload=payload))

        # Mock the default reviewers call
        reviewers_response = {"values": [{"uuid": "{reviewer-uuid}"}]}
        mock_client.get = AsyncMock(return_value=_response(json_payload=reviewers_response))

        with patch("koboi.tools.builtin.bitbucket.httpx.AsyncClient", return_value=mock_client):
            await BitbucketClient(username="user", app_password="token").create_pr(
                workspace="ws",
                repo_slug="repo",
                source_branch="feat",
                dest_branch="main",
                title="Title",
                description="Body",
                default_reviewers=True,
            )

        # Verify reviewers were fetched and included
        assert mock_client.get.call_count >= 1
        _, kwargs = mock_client.post.call_args
        assert "reviewers" in kwargs["json"]

    async def test_update_pr_omits_unset_fields(self):
        mock_client = _mock_async_client(_response(json_payload={"id": "1", "state": "open"}))
        with patch("koboi.tools.builtin.bitbucket.httpx.AsyncClient", return_value=mock_client):
            await BitbucketClient(username="user", app_password="token").update_pr(
                workspace="ws", repo_slug="repo", pr_id=1, title="New Title"
            )
        _, kwargs = mock_client.put.call_args
        assert kwargs["json"] == {"title": "New Title"}

    async def test_list_prs_returns_list(self):
        payload = {
            "values": [
                {"id": 1, "title": "A", "state": "OPEN", "links": {"html": {"href": "url1"}}},
                {"id": 2, "title": "B", "state": "OPEN", "links": {"html": {"href": "url2"}}},
            ]
        }
        with patch(
            "koboi.tools.builtin.bitbucket.httpx.AsyncClient",
            return_value=_mock_async_client(_response(json_payload=payload)),
        ):
            result = await BitbucketClient(username="user", app_password="token").list_prs("ws", "repo")
        assert len(result) == 2
        assert result[0]["id"] == 1

    async def test_get_default_reviewers_returns_uuids(self):
        payload = {"values": [{"uuid": "{uuid-1}"}, {"uuid": "{uuid-2}"}]}
        with patch(
            "koboi.tools.builtin.bitbucket.httpx.AsyncClient",
            return_value=_mock_async_client(_response(json_payload=payload)),
        ):
            result = await BitbucketClient(username="user", app_password="token").get_default_reviewers(
                "ws", "repo"
            )
        assert len(result) == 2
        assert result[0] == {"uuid": "{uuid-1}"}

    async def test_http_error_raises(self):
        error_resp = httpx.Response(401, request=httpx.Request("GET", "https://api.bitbucket.org/2.0"))
        with patch(
            "koboi.tools.builtin.bitbucket.httpx.AsyncClient", return_value=_mock_async_client(error_resp)
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await BitbucketClient(username="user", app_password="token").get_pr("ws", "repo", 1)


class TestToolsMissingClient:
    async def test_create_pr_no_client_returns_error(self):
        result = await bitbucket_create_pr("ws", "repo", "feat", "main", "Title", "Body", _deps=None)
        assert result.startswith("Error: Bitbucket is not configured")

    async def test_list_prs_no_client_returns_error(self):
        result = await bitbucket_list_prs("ws", "repo", _deps={})
        assert result.startswith("Error: Bitbucket is not configured")


class TestToolsWithMockedClient:
    async def test_create_pr_happy_path(self):
        client = AsyncMock()
        client.create_pr = AsyncMock(
            return_value={
                "id": 7,
                "links": {"html": {"href": "https://bitbucket.org/ws/repo/pullrequests/7"}},
            }
        )
        result = await bitbucket_create_pr(
            "ws", "repo", "feat", "main", "Title", "Body", _deps={"bitbucket_client": client}
        )
        assert "Created PR #7" in result
        assert "pullrequests/7" in result

    async def test_create_pr_http_status_error_formatted(self):
        client = AsyncMock()
        resp = httpx.Response(
            422, text="validation failed", request=httpx.Request("POST", "https://api.bitbucket.org/2.0")
        )
        client.create_pr = AsyncMock(side_effect=httpx.HTTPStatusError("x", request=resp.request, response=resp))
        result = await bitbucket_create_pr(
            "ws", "repo", "feat", "main", "Title", "Body", _deps={"bitbucket_client": client}
        )
        assert result.startswith("Error: Bitbucket API returned 422")

    async def test_update_pr_happy_path(self):
        client = AsyncMock()
        client.update_pr = AsyncMock(return_value={"id": "3", "state": "OPEN", "links": {"html": {"href": "u"}}})
        result = await bitbucket_update_pr("ws", "repo", 3, title="New Title", _deps={"bitbucket_client": client})
        assert "Updated PR #3" in result

    async def test_list_prs_happy_path(self):
        client = AsyncMock()
        client.list_prs = AsyncMock(
            return_value=[
                {"id": 1, "title": "A", "state": "OPEN", "links": {"html": {"href": "url1"}}},
                {"id": 2, "title": "B", "state": "OPEN", "links": {"html": {"href": "url2"}}},
            ]
        )
        result = await bitbucket_list_prs("ws", "repo", _deps={"bitbucket_client": client})
        assert "#1" in result
        assert "#2" in result

    async def test_list_prs_empty_returns_message(self):
        client = AsyncMock()
        client.list_prs = AsyncMock(return_value=[])
        result = await bitbucket_list_prs("ws", "repo", _deps={"bitbucket_client": client})
        assert "No open pull requests" in result

    async def test_get_pr_happy_path(self):
        client = AsyncMock()
        client.get_pr = AsyncMock(
            return_value={
                "id": 5,
                "title": "Test PR",
                "state": "OPEN",
                "author": {"display_name": "John Doe"},
                "source": {"branch": {"name": "feat"}},
                "destination": {"branch": {"name": "main"}},
                "description": "Test",
                "links": {"html": {"href": "url5"}},
            }
        )
        result = await bitbucket_get_pr("ws", "repo", 5, _deps={"bitbucket_client": client})
        assert "#5" in result
        assert "Test PR" in result
        assert "author=John Doe" in result

    async def test_get_default_reviewers_happy_path(self):
        client = AsyncMock()
        client.get_default_reviewers = AsyncMock(return_value=[{"uuid": "{uuid-1}"}, {"uuid": "{uuid-2}"}])
        result = await bitbucket_get_default_reviewers("ws", "repo", _deps={"bitbucket_client": client})
        assert "Default reviewers for ws/repo" in result
        assert "{uuid-1}" in result
        assert "{uuid-2}" in result

    async def test_get_default_reviewers_empty_returns_message(self):
        client = AsyncMock()
        client.get_default_reviewers = AsyncMock(return_value=[])
        result = await bitbucket_get_default_reviewers("ws", "repo", _deps={"bitbucket_client": client})
        assert "No default reviewers configured" in result

    async def test_invalid_state_validated_in_list_prs(self):
        client = AsyncMock()
        result = await bitbucket_list_prs("ws", "repo", state="invalid", _deps={"bitbucket_client": client})
        assert "Error: state must be one of" in result

    async def test_create_pr_with_close_source_branch(self):
        client = AsyncMock()
        client.create_pr = AsyncMock(
            return_value={
                "id": 10,
                "links": {"html": {"href": "https://bitbucket.org/ws/repo/pullrequests/10"}},
            }
        )
        result = await bitbucket_create_pr(
            "ws",
            "repo",
            "feat",
            "main",
            "Title",
            "Body",
            close_source_branch=True,
            _deps={"bitbucket_client": client},
        )
        assert "Created PR #10" in result
        # Verify the close_source_branch parameter was passed
        call_args = client.create_pr.call_args
        # Parameters are passed as positional args in our implementation
        assert call_args.args[-1] is True  # close_source_branch is the last positional param

    async def test_create_pr_without_default_reviewers(self):
        client = AsyncMock()
        client.create_pr = AsyncMock(
            return_value={
                "id": 11,
                "links": {"html": {"href": "https://bitbucket.org/ws/repo/pullrequests/11"}},
            }
        )
        result = await bitbucket_create_pr(
            "ws",
            "repo",
            "feat",
            "main",
            "Title",
            "Body",
            default_reviewers=False,
            _deps={"bitbucket_client": client},
        )
        assert "Created PR #11" in result
        call_args = client.create_pr.call_args
        # Parameters are passed as positional args in our implementation
        assert call_args.args[-2] is False  # default_reviewers is second-to-last positional param
