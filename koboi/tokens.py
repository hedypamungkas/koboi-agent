from __future__ import annotations

import json


def estimate_single(msg: dict) -> int:
    total = 0
    for val in msg.values():
        if isinstance(val, str):
            total += len(val)
        elif isinstance(val, (list, dict)):
            total += len(json.dumps(val, ensure_ascii=False))
        elif val:
            total += len(str(val))
    return max(total // 3, 1)


def estimate_tokens(messages: list[dict]) -> int:
    return sum(estimate_single(m) for m in messages)


# ---------------------------------------------------------------------------
# Optional real tokenizer (issue #3) -- tiktoken BPE for OpenAI models, with
# the chars/3 heuristic above as the universal fallback. tiktoken is an optional
# extra (``pip install koboi-agent[tokenizer]``); absent -> heuristic. Only the
# OpenAI provider gets a BPE counter (its encodings are public and accurate);
# Anthropic/Cloudflare/unknown keep the heuristic (no offline tokenizer).
# ---------------------------------------------------------------------------


def _pick_encoding(model: str | None):
    """Return a tiktoken Encoding for the model, else None.

    Tries the model-specific encoding, then falls back to o200k_base (current
    OpenAI default). Any failure -> None (caller falls back to heuristic).
    """
    import tiktoken  # local import; tiktoken is an optional extra

    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:  # nosec B110 - best-effort: unknown model falls back to o200k_base
            pass
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return None


def make_tokenizer(provider: str | None = None, model: str | None = None):
    """Return a ``messages -> int`` token counter using tiktoken BPE, or None.

    Returns None (so callers fall back to the heuristic) when:
      * tiktoken is not installed, or
      * the provider is not OpenAI (no accurate offline encoding available).

    The returned counter adds a small per-message framing overhead (~4 tokens)
    to better approximate real API prompt-token counts.
    """
    if not provider or str(provider).lower() != "openai":
        return None
    try:
        import tiktoken  # noqa: F401 -- imported to confirm availability
    except ImportError:
        return None
    enc = _pick_encoding(model)
    if enc is None:
        return None

    def _count(messages: list[dict]) -> int:
        total = 0
        for m in messages:
            for v in m.values():
                if isinstance(v, str):
                    total += len(enc.encode(v))
                elif isinstance(v, (list, dict)):
                    total += len(enc.encode(json.dumps(v, ensure_ascii=False)))
                elif v:
                    total += len(enc.encode(str(v)))
            total += 4  # per-message framing (role tags / delimiters)
        return max(total, 1)

    return _count
