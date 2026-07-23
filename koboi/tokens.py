from __future__ import annotations

import json

# Wave 3: chars-per-token divisor for the CODE-calibrated heuristic. Code
# tokenizes denser than prose (symbols, indentation, camelCase), so the
# non-OpenAI provider counter uses ~2.5 chars/token instead of 3 -- a
# conservative over-estimate that keeps context budgeting safe for coding
# agents on providers with no offline tokenizer.
CODE_CHARS_PER_TOKEN = 2.5


def estimate_single(msg: dict, divisor: float = 3) -> int:
    total = 0
    for val in msg.values():
        if isinstance(val, str):
            total += len(val)
        elif isinstance(val, (list, dict)):
            total += len(json.dumps(val, ensure_ascii=False))
        elif val:
            total += len(str(val))
    return max(int(total / divisor), 1)


def estimate_tokens(messages: list[dict], divisor: float = 3) -> int:
    return sum(estimate_single(m, divisor) for m in messages)


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


def _heuristic_counter(divisor: float):
    """A ``messages -> int`` counter using the chars/divisor heuristic + framing."""

    def _count(messages: list[dict]) -> int:
        total = sum(estimate_single(m, divisor) for m in messages)
        total += 4 * len(messages)  # per-message framing (role tags / delimiters)
        return max(total, 1)

    return _count


def make_tokenizer(provider: str | None = None, model: str | None = None):
    """Return a ``messages -> int`` token counter for the provider.

    OpenAI + tiktoken installed -> accurate BPE counter (unchanged).
    Any OTHER known provider (Anthropic/Cloudflare/...) -> a code-calibrated
    conservative heuristic (chars/2.5 + framing) instead of the old ``None``:
    code tokenizes denser than prose, and chars/3 under-estimated non-OpenAI
    prompts until the first real usage arrived (Wave 3 fidelity fix). The
    returned counter feeds ``ContextManager._effective_tokens``; the real
    ``last_actual_tokens`` from API usage still self-corrects after turn 1.

    Returns None only when no provider is given (callers keep the bare
    ``estimate_tokens`` chars/3 fallback).
    """
    if not provider:
        return None
    if str(provider).lower() != "openai":
        return _heuristic_counter(CODE_CHARS_PER_TOKEN)
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
