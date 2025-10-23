
import base64
import json
from typing import List, Dict, Optional

def decode_user_message(encoded_message: str) -> str:
    """
    Decode a Base64-encoded message payload to UTF-8 text.

    Parameters
    ----------
    encoded_message : str
        Base64 string. Leading/trailing whitespace is ignored.

    Returns
    -------
    str
        Decoded UTF-8 text. Returns "" on failure or empty input.
    """
    if not encoded_message:
        return ""
    try:
        return base64.b64decode(encoded_message.strip()).decode("utf-8")
    except Exception:
        # In production you might want to log the exception
        return ""


def decode_row_to_turns(encoded_b64: str) -> List[Dict[str, str]]:
    """
    Decode the Base64-encoded message and parse the 'turns' list.

    Parameters
    ----------
    encoded_b64 : str
        Base64-encoded string that wraps a JSON object like:
        {"turns": [{"role": "user", "text": "..."}, ...]}

    Returns
    -------
    List[Dict[str, str]]
        List of turn dicts. Returns [] on any failure.
    """
    decoded = decode_user_message(encoded_b64)
    if not decoded:
        return []
    try:
        obj = json.loads(decoded)
        return obj.get("turns", [])
    except Exception:
        return []
    

def _xml_escape(text: str) -> str:
    """Escape XML special characters."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
    )


def xml_transform(
    chat_id: str,
    timestamp: str,
    title: Optional[str],
    turns: List[Dict[str, str]],
) -> str:
    """
    Convert a single chat into an XML-like string suitable for LLM prompting,
    following the Clio paper's idea of role-tagged turns.

    Parameters
    ----------
    chat_id : str
        Unique conversation identifier.
    timestamp : str
        ISO-like timestamp (e.g., '2025-10-10T17:21').
    title : Optional[str]
        Short subject/title; may be None or "" if unavailable.
    turns : List[Dict[str, str]]
        A list of dicts like: {"role": "user"|"assistant", "text": "<message>"}.
        (If you have additional fields they will be ignored.)

    Returns
    -------
    str
        XML-like representation, for example:

        <conversation id="...">
          <meta>
            <timestamp>...</timestamp>
            <title>...</title>
          </meta>
          <turn role="user">Hello</turn>
          <turn role="assistant">Hi there</turn>
        </conversation>
    """
    # Build header
    parts = [f'<conversation id="{_xml_escape(chat_id)}">']
    parts.append("  <meta>")
    parts.append(f"    <timestamp>{_xml_escape(timestamp)}</timestamp>")
    if title:
        parts.append(f"    <title>{_xml_escape(title)}</title>")
    parts.append("  </meta>")

    # Turns
    for t in turns or []:
        role = _xml_escape(str(t.get("role", "")))
        text = _xml_escape(str(t.get("text", "")))
        parts.append(f'  <turn role="{role}">{text}</turn>')

    parts.append("</conversation>")
    return "\n".join(parts)
