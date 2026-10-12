"""Parse .eml, .mbox, and JSON files into Email objects."""

from __future__ import annotations

import email
import email.policy
import email.utils
import json
import mailbox
from datetime import datetime, timezone
from pathlib import Path

from ekb.models import Email


def parse_file(path: str | Path) -> list[Email]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".eml":
        return [_parse_eml(path.read_bytes())]
    if suffix == ".json":
        return _parse_json(path)
    if suffix == ".mbox":
        return _parse_mbox(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _parse_eml(raw: bytes) -> Email:
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    body = _extract_body(msg)
    return _msg_to_email(msg, body)


def _parse_json(path: Path) -> list[Email]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    return [_dict_to_email(d) for d in data]


def _parse_mbox(path: Path) -> list[Email]:
    mbox = mailbox.mbox(str(path))
    results = []
    for msg in mbox:
        body = _extract_body(msg)
        results.append(_msg_to_email(msg, body))
    return results


def _extract_body(msg: email.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="replace")
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="replace")
    payload = msg.get_payload(decode=True)
    if payload:
        return payload.decode("utf-8", errors="replace")
    return str(msg.get_payload())


def _parse_date(raw: str | None) -> datetime:
    if not raw:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    parsed = email.utils.parsedate_to_datetime(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_recipients(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [addr for _, addr in email.utils.getaddresses([raw]) if addr]


def _msg_to_email(msg: email.message.Message, body: str) -> Email:
    refs_raw = msg.get("References", "")
    refs = refs_raw.split() if refs_raw else []
    _, sender_addr = email.utils.parseaddr(msg.get("From", ""))
    return Email(
        message_id=msg.get("Message-ID", ""),
        subject=msg.get("Subject", ""),
        sender=sender_addr or msg.get("From", ""),
        recipients=_parse_recipients(msg.get("To")),
        date=_parse_date(msg.get("Date")),
        body=body,
        in_reply_to=msg.get("In-Reply-To"),
        references=refs,
        headers={k: str(v) for k, v in msg.items()},
    )


def _dict_to_email(d: dict) -> Email:
    date_raw = d.get("date", "")
    if isinstance(date_raw, str):
        try:
            dt = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
        except ValueError:
            dt = _parse_date(date_raw)
    else:
        dt = datetime(1970, 1, 1, tzinfo=timezone.utc)

    to = d.get("to", [])
    if isinstance(to, str):
        to = [to]

    return Email(
        message_id=d.get("message_id", ""),
        subject=d.get("subject", ""),
        sender=d.get("from", ""),
        recipients=to,
        date=dt,
        body=d.get("body", ""),
        in_reply_to=d.get("in_reply_to"),
        references=d.get("references", []),
    )
