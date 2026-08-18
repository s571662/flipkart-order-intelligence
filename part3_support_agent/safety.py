import re


INJECTION_PATTERNS = [
    r"ignore (all|any|the|previous) instructions",
    r"ignore previous",
    r"disregard (all|any|the|previous) instructions",
    r"override (the )?(system|rules|instructions)",
    r"reveal (the )?(system prompt|hidden prompt|instructions)",
    r"pretend you are",
    r"act as if",
]


def detect_prompt_injection(user_text: str) -> dict:
    normalized = user_text.lower().strip()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized):
            return {
                "blocked": True,
                "reason": "Potential prompt injection detected.",
                "matched_pattern": pattern,
            }

    return {
        "blocked": False,
        "reason": None,
        "matched_pattern": None,
    }