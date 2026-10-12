"""Strip quoted reply text and signatures from email bodies."""

from __future__ import annotations

import re

# "On <date>, <name> wrote:" — Gmail/Apple Mail style
_QUOTE_HEADER = re.compile(
    r"^On .{10,80} wrote:\s*$", re.MULTILINE
)

# Outlook-style forwarded/reply header block
_OUTLOOK_HEADER = re.compile(
    r"^-{2,}\s*(Original Message|Forwarded message)\s*-{2,}\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Signature delimiter per RFC 3676
_SIG_DELIM = re.compile(r"^-- ?\s*$", re.MULTILINE)

# Common sign-off lines that precede a name-only signature
_SIGNOFF = re.compile(
    r"^(regards|cheers|thanks|best|sincerely|- \w+)\s*,?\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def strip_quotes(body: str) -> str:
    """Remove quoted reply blocks, keeping only the author's new content."""
    # Cut at "On ... wrote:" header
    m = _QUOTE_HEADER.search(body)
    if m:
        body = body[: m.start()].rstrip()

    # Cut at Outlook-style header
    m = _OUTLOOK_HEADER.search(body)
    if m:
        body = body[: m.start()].rstrip()

    # Remove line-level quoting (lines starting with >)
    lines = body.split("\n")
    cleaned = [l for l in lines if not l.startswith(">")]
    return "\n".join(cleaned).strip()


def strip_signature(body: str) -> str:
    """Remove email signature block."""
    # RFC 3676 sig delimiter
    m = _SIG_DELIM.search(body)
    if m:
        return body[: m.start()].rstrip()

    # Fallback: common sign-off as last non-empty content
    lines = body.rstrip().split("\n")
    for i in range(len(lines) - 1, max(len(lines) - 6, -1), -1):
        if _SIGNOFF.match(lines[i].strip()):
            # Keep everything before the sign-off only if there's
            # substantial content above it
            above = "\n".join(lines[:i]).strip()
            if len(above) > 20:
                return above
            break
    return body


def clean(body: str) -> str:
    """Full cleaning pass: strip quotes then signature."""
    return strip_signature(strip_quotes(body))
