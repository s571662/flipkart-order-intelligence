def mock_llm_response(intent: str, context: dict) -> dict:
    if intent == "policy":
        return {
            "answer": context.get(
                "answer",
                "I do not have enough policy evidence to answer."
            ),
            "source": "mock_llm",
        }

    if intent == "return_risk":
        probability = context.get("return_probability", 0.0)
        risk_bucket = context.get("risk_bucket", "Unknown")

        return {
            "answer": (
                f"Return probability is {probability:.2%}. "
                f"Risk bucket: {risk_bucket}."
            ),
            "source": "mock_llm",
        }

    if intent == "image":
        product_class = context.get(
            "product_class",
            "Unknown"
        )
        confidence = context.get(
            "confidence",
            0.0
        )

        return {
            "answer": (
                f"Predicted product category: {product_class}. "
                f"Confidence: {confidence:.2%}."
            ),
            "source": "mock_llm",
        }

    if intent == "order_context":
        order_id = context.get(
            "order_id",
            "Unknown"
        )

        answer = context.get(
            "answer",
            f"You are asking about order {order_id}."
        )

        return {
            "answer": answer,
            "source": "mock_llm",
        }

    return {
        "answer": (
            "I can help with return-risk prediction, "
            "product-image classification, or return-policy questions."
        ),
        "source": "mock_llm",
    }