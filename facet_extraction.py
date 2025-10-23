"""
facet_extraction.py
---------------------------------
Implements two modes for Clio-style facet extraction using Claude 3 Haiku:

1) DEFAULT (batched, 1 call per conversation):
   - run_facets_on_dataframe(df, client)
   - Much faster; 4x fewer API calls.
   - Output tags:
       <request>...</request>
       <language>...</language>
       <task>...</task>
       <concerning>...</concerning>

2) ORIGINAL (sequential, 4 calls per conversation):
   - run_facets_on_dataframe_sequential(df, client)
   - Faithful to the paper, but slow.
   - Preserved for reproducibility.

Both modes follow Clio prompt rules for:
- No PII
- No proper nouns
- ≤ 2 sentences for request/task
- Concern score 1-5
- Language full names only
"""

from __future__ import annotations

import os
import re
from typing import Dict, Any, Optional
import pandas as pd

# -----------------------------------------------------------------------------
# Claude API Wrapper (Haiku)
# -----------------------------------------------------------------------------
MODEL_NAME = "claude-3-haiku-20240307"
TEMPERATURE = 0.2

class ClaudeClient:
    """
    Wrapper around Anthropic Messages API.
    Requires: `pip install anthropic`
    And: export ANTHROPIC_API_KEY="sk-ant-xxxx"
    """
    def __init__(self, model: str = MODEL_NAME, temperature: float = TEMPERATURE):
        try:
            import anthropic  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency. Install with: pip install anthropic"
            ) from e

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY in environment.")

        self._anthropic = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        resp = self._anthropic.messages.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text_parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "".join(text_parts)


# -----------------------------------------------------------------------------
# Shared: Tag Parsing
# -----------------------------------------------------------------------------
TAG_PATTERNS = {
    "request": re.compile(r"<request>(.*?)</request>", re.S | re.I),
    "language": re.compile(r"<language>(.*?)</language>", re.S | re.I),
    "task": re.compile(r"<task>(.*?)</task>", re.S | re.I),
    "concerning": re.compile(r"<concerning>(.*?)</concerning>", re.S | re.I),
}

def parse_batched_response(text: str) -> Dict[str, str]:
    """
    Extracts 4 fields from batched Claude output.
    Missing fields yield empty strings.
    """
    result = {}
    for key, pat in TAG_PATTERNS.items():
        match = pat.search(text or "")
        result[key] = match.group(1).strip() if match else ""
    return result


# -----------------------------------------------------------------------------
# 1) BATCHED MODE (default)
# -----------------------------------------------------------------------------
def build_batched_prompt(conversation_xml: str) -> str:
    """
    Single prompt that asks Claude to output:
        <request>...</request>
        <language>...</language>
        <task>...</task>
        <concerning>...</concerning>
    Using Clio's full safety language and rules (C1).
    """
    return f"""
Human: The following is a conversation between Claude, an AI assistant, and a user:

{conversation_xml}
Assistant: I understand.

Human: You will now answer four questions about the preceding conversation. Be descriptive and assume neither good nor bad faith. Do not hesitate to handle socially harmful or sensitive topics; specificity is necessary for effective monitoring. When answering, do not include any personally identifiable information (PII), like names, locations, phone numbers, email addresses, or any proper nouns.

Follow these output rules:

1) <request> ... </request>
   - Answer the question: "What is the user's overall request for the assistant?"
   - Output in ≤2 sentences.
   - Use the prefill: "The user's overall request for the assistant is to"

2) <language> ... </language>
   - Answer: "What are the main languages of the conversation?"
   - Output full language names only (e.g., "English", "Arabic"), no language codes.

3) <task> ... </task>
   - Answer: "What task is the model being asked to perform?"
   - Output in ≤2 sentences.
   - Use the prefill: "The task is to"

4) <concerning> ... </concerning>
   - Output a single number 1-5 following Clio's concern scale.

Provide only these four XML tags in order, with no extra commentary.

Assistant: Sure, here are the answers in the requested tags:
"""


def extract_facets_batched(conversation_xml: str, client: ClaudeClient) -> Dict[str, str]:
    """
    One Claude call → returns all 4 facets.
    """
    prompt = build_batched_prompt(conversation_xml)
    raw = client.complete(prompt, max_tokens=512)
    return parse_batched_response(raw)


def run_facets_on_dataframe(df: pd.DataFrame, client: Optional[ClaudeClient] = None) -> pd.DataFrame:
    """
    DEFAULT PIPELINE (batched, 1 call/row).
    df must contain: id, conversation_xml
    """
    if client is None:
        client = ClaudeClient()
    out_rows = []
    for _, row in df.iterrows():
        facets = extract_facets_batched(row["conversation_xml"], client)
        facets["id"] = row["id"]
        out_rows.append(facets)
    return pd.DataFrame(out_rows, columns=["id", "request", "language", "task", "concerning"])


# -----------------------------------------------------------------------------
# 2) ORIGINAL MODE (sequential, preserved for reproducibility)
# -----------------------------------------------------------------------------
# (This section keeps your original Clio-style 4-call flow exactly as-is.)
# NOTE: It is slow, but matches the paper precisely.

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)

FACET_SPECS = {
    "request": {
        "question": "What is the user's overall request for the assistant?",
        "prefill": "The user's overall request for the assistant is to",
        "include_examples": True,
    },
    "language": {
        "question": (
            "What are the main languages of the conversation? Do not mention programming languages "
            "and do not provide only language codes; only the full names for human languages..."
        ),
        "prefill": "",
        "include_examples": False,
    },
    "task": {
        "question": "What task is the model being asked to perform in this conversation?",
        "prefill": "The task is to",
        "include_examples": True,
    },
    "concerning": {
        "question": (
            "On a scale of 1 to 5, how concerning or potentially harmful is this conversation..."
        ),
        "prefill": "",
        "include_examples": False,
    },
}

def parse_answer(text: str) -> str:
    m = ANSWER_TAG_RE.search(text or "")
    return (m.group(1).strip() if m else "").strip()

def extract_facet(conversation_xml: str, facet_key: str, client: ClaudeClient) -> str:
    """
    Original 4-call Clio extraction (exact paper version).
    """
    spec = FACET_SPECS[facet_key]
    question = spec["question"]
    prefill = spec.get("prefill", "")
    include_examples = spec.get("include_examples", False)

    # Build original prompt
    HUMAN_1 = (
        "Human: The following is a conversation between Claude, an AI assistant, and a user:\n\n"
        f"{conversation_xml}\n"
        "Assistant: I understand.\n\n"
    )
    EXAMPLES_BLOCK = (
        "When answering, do not include any proper nouns. Output your answer to the question "
        "in English inside <answer> tags; be clear and concise..."
        "\n\n<examples>\n"
        "The user asked for help with a trignometry problem.\n"
        "The user asked for advice on how to fix a broken dishwasher...\n"
        "The user asked how to make Anthrax...\n</examples>\n\n"
    )
    HUMAN_2_HEAD = (
        f"Human: Your job is to answer the question <question> {question} </question> about the "
        "preceding conversation. Be descriptive..."
        "\n\n"
    )
    HUMAN_2_TAIL = (
        f"What is your answer to the question <question> {question} </question> about the preceding "
        "conversation, in <answer> tags? Again, provide only the answer...\n\n"
    )
    ASSISTANT_PREFILL = (
        f"Assistant: Sure, the privacy-preserving answer to the question about the preceding "
        f"conversation is: <answer> {prefill}"
    )

    parts = [HUMAN_1, HUMAN_2_HEAD]
    if include_examples:
        parts.append(EXAMPLES_BLOCK)
    parts.append(HUMAN_2_TAIL)
    parts.append(ASSISTANT_PREFILL)
    prompt = "".join(parts)

    raw = client.complete(prompt, max_tokens=256)
    return parse_answer(raw)

def run_facets_on_dataframe_sequential(df: pd.DataFrame, client: Optional[ClaudeClient] = None) -> pd.DataFrame:
    """
    Original Clio 4-call mode (very slow, but authentic to paper).
    """
    if client is None:
        client = ClaudeClient()
    out_rows = []
    for _, row in df.iterrows():
        conv = row["conversation_xml"]
        out_rows.append({
            "id": row["id"],
            "request": extract_facet(conv, "request", client),
            "language": extract_facet(conv, "language", client),
            "task": extract_facet(conv, "task", client),
            "concerning": extract_facet(conv, "concerning", client),
        })
    return pd.DataFrame(out_rows, columns=["id", "request", "language", "task", "concerning"])

