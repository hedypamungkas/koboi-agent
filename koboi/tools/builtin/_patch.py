"""koboi/tools/builtin/_patch -- Unified-diff parser + applier for apply_patch.

Hand-rolled (there is no stdlib patch *applier*; ``difflib`` only generates diffs,
and adding a third-party dep would break the minimal-core rule). Parses the
unified-diff grammar that ``git diff`` / ``diff -u`` emit (``---``/``+++`` file
headers, ``@@`` hunk headers, context/``+``/``-`` lines, ``\\ No newline at end of
file``) into old/new text blocks, then applies them by exact-substring match --
reusing ``edit_file``'s count-for-uniqueness semantics (content-based, so line
drift between the diff's ``@@`` line numbers and the live file is tolerated).

All-or-nothing: ``apply_hunks`` raises ``PatchError`` (naming the failing hunk)
if any hunk fails to match, leaving the caller's content unmutated so the tool
can decline the atomic write.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class PatchError(ValueError):
    """Malformed patch, or a hunk that failed to apply.

    The message always names which hunk (#N) failed and why, so the model can
    re-read the file and regenerate the diff against current content.
    """


# ``@@ -old_start[,old_count] +new_start[,new_count] @@ [optional section]``
# Counts default to 1 when omitted (the ``diff`` convention for single-line hunks).
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Hunk:
    """One unified-diff hunk resolved to its old/new text blocks."""

    old_start: int  # 1-based source line from ``@@ -N`` (advisory; error messages only)
    new_start: int  # 1-based target line from ``@@ +N`` (advisory)
    old_text: str  # context + removed lines joined (the search target)
    new_text: str  # context + added lines joined (the replacement)


def _strip_trailing_newline(buf: list[str]) -> None:
    """Strip exactly one trailing ``\\n`` from the last entry of ``buf`` (in place).

    Used to honor a ``\\ No newline at end of file`` marker on the preceding line.
    """
    if buf and buf[-1].endswith("\n"):
        buf[-1] = buf[-1][:-1]


def parse_unified_diff(patch: str) -> list[Hunk]:
    """Parse a unified-diff ``patch`` string into a list of :class:`Hunk`.

    Accepts bare hunks (``@@ ...``) or a leading ``---``/``+++`` file-header pair
    and ``diff --git``/``index`` prologue lines, all of which are skipped --
    ``apply_patch`` targets a single explicit ``path`` arg. Raises
    :class:`PatchError` on malformed input (empty patch, bad hunk header, unknown
    line marker).
    """
    if not patch or not patch.strip():
        raise PatchError("patch is empty")
    lines = patch.splitlines(keepends=True)
    hunks: list[Hunk] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        # Tolerated (and ignored) prologue: git header lines and a --- / +++ pair.
        if line.startswith(("diff --git", "index ", "--- ")):
            if line.startswith("--- ") and not (i + 1 < n and lines[i + 1].startswith("+++ ")):
                raise PatchError(f"malformed patch: '---' header without '+++' at line {i + 1}")
            i += 2 if line.startswith("--- ") else 1
            continue
        m = _HUNK_HEADER_RE.match(line.rstrip("\r\n"))
        if not m:
            raise PatchError(
                f"malformed patch: expected '@@ ... @@' hunk header at line {i + 1}, got {line.rstrip()!r}"
            )
        old_start = int(m.group(1))
        new_start = int(m.group(3))
        i += 1
        old_parts: list[str] = []
        new_parts: list[str] = []
        last_marker: str | None = None
        while i < n:
            body = lines[i]
            if body.startswith("@@"):
                break  # next hunk follows immediately
            if body.startswith(("--- ", "diff --git", "index ")):
                break  # prologue of a (tolerated) second file header ends this hunk
            marker = body[:1]
            if marker == "\\":
                # ``\ No newline at end of file`` refers to the immediately preceding
                # CONTENT line; strip its trailing newline from the side(s) it lived on.
                if last_marker == " ":
                    _strip_trailing_newline(old_parts)
                    _strip_trailing_newline(new_parts)
                elif last_marker == "-":
                    _strip_trailing_newline(old_parts)
                elif last_marker == "+":
                    _strip_trailing_newline(new_parts)
                i += 1
                continue
            if marker == " ":
                rest = body[1:]
                old_parts.append(rest)
                new_parts.append(rest)
            elif marker == "-":
                old_parts.append(body[1:])
            elif marker == "+":
                new_parts.append(body[1:])
            elif body in ("\n", "\r\n"):
                # Recover a blank context line whose leading space got stripped to a
                # bare newline (editors/transport stripping trailing whitespace) --
                # treat it as an empty context line.
                old_parts.append(body)
                new_parts.append(body)
                last_marker = " "
                i += 1
                continue
            else:
                raise PatchError(f"malformed patch: unexpected line marker {marker!r} at line {i + 1}")
            last_marker = marker
            i += 1
        hunks.append(
            Hunk(
                old_start=old_start,
                new_start=new_start,
                old_text="".join(old_parts),
                new_text="".join(new_parts),
            )
        )
    if not hunks:
        raise PatchError("patch contained no valid hunks")
    return hunks


def apply_hunks(content: str, hunks: list[Hunk]) -> str:
    """Apply ``hunks`` to ``content`` and return the new content.

    Each hunk's ``old_text`` must occur exactly once in the progressively-mutated
    content (exact-substring match, reusing ``edit_file``'s uniqueness rule).
    Raises :class:`PatchError` on a missing or ambiguous match; ``content`` is
    never partially returned (the caller writes atomically only on full success).
    """
    for idx, hunk in enumerate(hunks, start=1):
        if hunk.old_text == hunk.new_text:
            continue  # no-op hunk (context-only or identical add/remove)
        if not hunk.old_text:
            raise PatchError(
                f"hunk #{idx} is a pure insertion with no context -- apply_patch "
                "needs at least one surrounding context line to place the edit "
                "(use write_file to create a new file)."
            )
        count = content.count(hunk.old_text)
        if count == 0:
            raise PatchError(
                f"hunk #{idx} did not apply: context not found in file "
                f"(expected near line {hunk.old_start}). Re-read the file and "
                "regenerate the diff against the current content."
            )
        if count > 1:
            raise PatchError(
                f"hunk #{idx} did not apply: context matched {count} times -- "
                "include more surrounding context lines so it is unique."
            )
        content = content.replace(hunk.old_text, hunk.new_text, 1)
    return content
