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
    """Strip exactly one trailing line ending (``\\n`` or ``\\r\\n``) from the last
    entry of ``buf`` (in place).

    Used to honor a ``\\ No newline at end of file`` marker on the preceding line.
    Strips the full CRLF pair so a CRLF patch does not leave a stray ``\\r`` that
    would then fail to match an LF file.
    """
    if buf and buf[-1].endswith("\n"):
        buf[-1] = buf[-1][:-1]
        if buf[-1].endswith("\r"):
            buf[-1] = buf[-1][:-1]


def _patch_filename(header: str) -> str:
    """Extract the filename from a ``--- ``/``+++ `` header line.

    Strips the 4-char marker, an optional ``a/``/``b/`` VCS prefix, surrounding
    quotes, and any tab-separated timestamp -- so ``+++ b/foo.py\\t2024-01-01``
    yields ``foo.py``.
    """
    name = header[4:].rstrip("\r\n")
    if name.startswith(("a/", "b/")):
        name = name[2:]
    return name.split("\t", 1)[0].strip().strip('"')


def parse_unified_diff(patch: str) -> list[Hunk]:
    """Parse a unified-diff ``patch`` string into a list of :class:`Hunk`.

    Accepts bare hunks (``@@ ...``) or a leading ``---``/``+++`` file-header pair
    and ``diff --git``/``index`` prologue lines, all of which are skipped --
    ``apply_patch`` targets a single explicit ``path`` arg. Raises
    :class:`PatchError` on malformed input (empty patch, bad hunk header, unknown
    line marker, a multi-file patch, or a ``\\``-prefixed line that is not the
    exact ``\\ No newline at end of file`` marker).

    Hunk bodies are read COUNT-based (the ``@@ -o,oc +n,nc @@`` old/new counts)
    so a content line that happens to start with ``--- ``/``+++ `` (a markdown
    rule, a removed line whose content begins with ``-``) is NOT mistaken for a
    file header -- the prior marker-scan terminated hunks early on such lines.
    """
    if not patch or not patch.strip():
        raise PatchError("patch is empty")
    lines = patch.splitlines(keepends=True)
    hunks: list[Hunk] = []
    first_target: str | None = None
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        # Tolerated (and ignored) prologue: git header lines + --- / +++ pairs.
        if line.startswith(("diff --git", "index ")):
            i += 1
            continue
        if line.startswith("--- "):
            if not (i + 1 < n and lines[i + 1].startswith("+++ ")):
                raise PatchError(f"malformed patch: '---' header without '+++' at line {i + 1}")
            target = _patch_filename(lines[i + 1])
            if first_target is None:
                first_target = target
            elif target != first_target:
                # apply_patch edits ONE explicit path; a multi-file patch would
                # silently flatten the second file's hunks onto the first.
                raise PatchError(
                    f"malformed patch: apply_patch targets a single file, but this patch "
                    f"also touches {target!r} (first file {first_target!r}). Apply one "
                    "patch per file."
                )
            i += 2
            continue
        if line.startswith("+++ "):
            # Bare +++ without a preceding --- (some emitters) -- tolerate as prologue.
            i += 1
            continue
        m = _HUNK_HEADER_RE.match(line.rstrip("\r\n"))
        if not m:
            raise PatchError(
                f"malformed patch: expected '@@ ... @@' hunk header at line {i + 1}, got {line.rstrip()!r}"
            )
        old_start = int(m.group(1))
        old_count = int(m.group(2)) if m.group(2) is not None else 1
        new_start = int(m.group(3))
        new_count = int(m.group(4)) if m.group(4) is not None else 1
        counts_explicit = m.group(2) is not None or m.group(4) is not None
        i += 1
        old_parts: list[str] = []
        new_parts: list[str] = []
        last_marker: str | None = None
        old_left, new_left = old_count, new_count
        while i < n:
            body = lines[i]
            # Termination. With explicit counts, the hunk ends when both counts
            # are satisfied (then drain trailing no-newline markers). With omitted
            # counts (``diff`` drops the ,count only for true 1-line hunks, but a
            # hand-crafted/model patch may be inconsistent) we fall back to a
            # structural boundary -- so a content line starting with ``--- `` is
            # only a terminator when it opens a real ``--- ``/``+++ `` file pair.
            if counts_explicit and old_left <= 0 and new_left <= 0:
                while i < n and lines[i].startswith("\\"):
                    mk = lines[i]
                    if mk.rstrip("\r\n") != "\\ No newline at end of file":
                        raise PatchError(f"malformed patch: unexpected line marker {mk.rstrip()!r} at line {i + 1}")
                    if last_marker == " ":
                        _strip_trailing_newline(old_parts)
                        _strip_trailing_newline(new_parts)
                    elif last_marker == "-":
                        _strip_trailing_newline(old_parts)
                    elif last_marker == "+":
                        _strip_trailing_newline(new_parts)
                    else:
                        raise PatchError(
                            f"malformed patch: '\\ No newline' marker with no preceding content line at line {i + 1}"
                        )
                    i += 1
                break
            if not counts_explicit:
                if body.startswith("@@") or body.startswith(("diff --git", "index ")):
                    break
                if body.startswith("--- ") and i + 1 < n and lines[i + 1].startswith("+++ "):
                    break  # a genuine file-header pair opens a new file
            elif body.startswith("@@"):
                # Explicit counts claimed more lines than present.
                raise PatchError(
                    f"malformed patch: hunk header @@ -{old_start},{old_count} "
                    f"+{new_start},{new_count} declared more lines than present "
                    f"(hit next '@@' at line {i + 1})"
                )
            marker = body[:1]
            if marker == "\\":
                # Only the EXACT ``\ No newline at end of file`` marker is valid;
                # any other backslash-prefixed line is content the model wrote (a
                # regex like ``\bword``) and must NOT be silently dropped.
                if body.rstrip("\r\n") != "\\ No newline at end of file":
                    raise PatchError(f"malformed patch: unexpected line marker {body.rstrip()!r} at line {i + 1}")
                if last_marker is None:
                    raise PatchError(
                        f"malformed patch: '\\ No newline' marker with no preceding content line at line {i + 1}"
                    )
                # The marker refers to the immediately preceding CONTENT line; strip
                # its trailing newline from the side(s) it lived on.
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
                old_left -= 1
                new_left -= 1
            elif marker == "-":
                old_parts.append(body[1:])
                old_left -= 1
            elif marker == "+":
                new_parts.append(body[1:])
                new_left -= 1
            elif body in ("\n", "\r\n"):
                # Recover a blank context line whose leading space got stripped to a
                # bare newline (editors/transport stripping trailing whitespace).
                old_parts.append(body)
                new_parts.append(body)
                old_left -= 1
                new_left -= 1
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
