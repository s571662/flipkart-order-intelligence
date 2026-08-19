from typing import Optional


# ---------------------------------------------------------
# Fixed response schema required by the support assistant
# ---------------------------------------------------------

RESPONSE_SCHEMA = {
    "answer": "string",
    "source": (
        "policy_kb | "
        "return_risk_tool | "
        "image_classifier_tool"
    ),
    "confidence": "float between 0.0 and 1.0",
}


# ---------------------------------------------------------
# 4S Prompt Engineering
# ---------------------------------------------------------
#
# Specific:
#   Defines the exact support capabilities and grounding rules.
#
# Short:
#   Requires concise answers without unnecessary explanation.
#
# Surround:
#   Requires the model to use only the supplied context/tool result.
#
# Single:
#   Gives one output task: return one structured JSON object.
#
# Role:
#   Explicitly establishes the Flipkart customer-support role.
# ---------------------------------------------------------

SYSTEM_PROMPT = """
ROLE:
You are Flipkart's Order Intelligence and Support Assistant.

SPECIFIC:
Handle only these supported capabilities:
1. Answer return/refund/delivery-policy questions from the policy knowledge base.
2. Report return-risk results produced by the return-risk tool.
3. Report product categories produced by the image-classification tool.
Never invent policy facts, model predictions, product classes, or confidence scores.

SHORT:
Keep the answer concise and directly relevant to the customer's request.

SURROUND:
Use only information contained inside the supplied <context> block.
If the policy context is not sufficiently grounded, refuse to invent an answer.

SINGLE:
Return exactly one structured JSON response containing:
- answer
- source
- confidence

For supported capability responses, source must be exactly one of:
- policy_kb
- return_risk_tool
- image_classifier_tool
""".strip()


# ---------------------------------------------------------
# Few-shot intent-routing examples
# ---------------------------------------------------------

INTENT_FEW_SHOTS = [
    {
        "user_query": "What is the return policy for electronics?",
        "intent": "policy",
    },
    {
        "user_query": "What is the return risk for this order?",
        "intent": "return_risk",
    },
    {
        "user_query": "Classify this product image.",
        "intent": "image",
    },
]


def normalize_query(text: str) -> str:
    return " ".join(
        text.lower().strip().split()
    )


def few_shot_intent(
    user_query: str,
) -> Optional[str]:
    """
    Deterministic MOCK_LLM routing support.

    Exact matches to the recorded few-shot examples return the
    demonstrated intent. Other queries fall back to the agent's
    deterministic routing rules.
    """
    normalized_query = normalize_query(
        user_query
    )

    for example in INTENT_FEW_SHOTS:
        if normalized_query == normalize_query(
            example["user_query"]
        ):
            return example["intent"]

    return None


def build_intent_prompt(
    user_query: str,
) -> str:
    """
    Build the deterministic intent-classification prompt representation.
    No external LLM or network call is performed.
    """
    examples = "\n".join(
        (
            f'User: "{example["user_query"]}" '
            f'-> Intent: {example["intent"]}'
        )
        for example in INTENT_FEW_SHOTS
    )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        "INTENT ROUTING EXAMPLES:\n"
        f"{examples}\n\n"
        f'CURRENT USER QUERY:\n"{user_query}"'
    )