"""tests/test_multi_provider_config.py -- Tier 2 (pools) wiring through the facade.

No network: client CONSTRUCTION doesn't call the wire (only complete() does), so
these tests build real ``ProviderPool`` instances from config and assert wiring
(member models, policy, breaker config, validation relaxation, embedding pool).
Failover BEHAVIOR is covered by tests/test_pool.py (FakeClient-driven).
"""

from __future__ import annotations

import pytest

from koboi.config import Config
from koboi.facade import _build_embedding_client, _resolve_chat_client
from koboi.llm.pool import FailoverPolicy, ProviderPool


def _cfg(yaml_text, validate=False):
    import yaml

    return Config(yaml.safe_load(yaml_text), validate=validate)


POOL_YAML = """
agent: { name: x }
providers:
  openai_a: { provider: openai, model: gpt-5.4, api_key: key-a, base_url: "http://a" }
  anthropic: { provider: anthropic, model: claude-sonnet-5, api_key: key-b, base_url: "http://b", auth_token: tok }
pools:
  resilient:
    providers: [openai_a, anthropic]
    policy: failover
    circuit_breaker: { failures: 5, cooldown_s: 60 }
llm: { pool: resilient }
"""


class TestChatPoolWiring:
    def test_resolve_chat_client_builds_pool(self):
        cfg = _cfg(POOL_YAML)
        client = _resolve_chat_client(cfg, logger=None)
        assert isinstance(client, ProviderPool)
        assert len(client.clients) == 2
        # Members built in declared order with their provider's model.
        assert client.clients[0].model == "gpt-5.4"
        assert client.clients[1].model == "claude-sonnet-5"
        # pool.model labels the first member (telemetry).
        assert client.model == "gpt-5.4"

    def test_pool_policy_and_breaker_config(self):
        cfg = _cfg(POOL_YAML)
        client = _resolve_chat_client(cfg, logger=None)
        assert isinstance(client._policy, FailoverPolicy)
        assert client.breaker.failure_threshold == 5
        assert client.breaker.cooldown_s == 60

    def test_unknown_pool_raises(self):
        cfg = _cfg("agent: { name: x }\nllm: { pool: missing }\n")
        with pytest.raises(ValueError, match="Unknown pool reference 'missing'"):
            _resolve_chat_client(cfg, logger=None)

    def test_inline_top_level_still_works(self):
        cfg = _cfg("agent: { name: x }\nllm: { provider: openai, model: gpt-5.4, api_key: k }\n")
        client = _resolve_chat_client(cfg, logger=None)
        assert not isinstance(client, ProviderPool)  # plain single client
        assert client.model == "gpt-5.4"

    def test_pool_spec_passes_strict_validation(self):
        """`llm: {pool: name}` must pass from_yaml validation (model requirement
        relaxed for pool refs)."""
        import yaml

        from koboi.config import Config as C

        data = yaml.safe_load(POOL_YAML)
        cfg = C(data, validate=True)  # would raise without the relaxation
        assert isinstance(_resolve_chat_client(cfg, logger=None), ProviderPool)


class TestEmbeddingPool:
    def test_embedding_pool_built(self):
        cfg = _cfg(
            """
            agent: { name: x }
            providers:
              emb_a: { provider: openai, model: text-embedding-3-small, api_key: ek-a, base_url: "http://ea" }
              emb_b: { provider: openai, model: text-embedding-3-small, api_key: ek-b, base_url: "http://eb" }
            pools:
              emb_pool: { providers: [emb_a, emb_b], policy: failover }
            embedding: { pool: emb_pool }
            """
        )
        emb = _build_embedding_client(cfg, logger=None)
        assert isinstance(emb, ProviderPool)
        assert len(emb.clients) == 2

    def test_embedding_inline_still_works(self):
        cfg = _cfg(
            "agent: { name: x }\n"
            "embedding: { provider: openai, model: text-embedding-3-small, api_key: ek, base_url: 'http://e' }\n"
        )
        emb = _build_embedding_client(cfg, logger=None)
        assert not isinstance(emb, ProviderPool)  # plain embedding client or None
