from part3_support_agent.agent import agent_graph


def test_policy_route():
    result = agent_graph.invoke({
        "user_query": "What is the return policy for electronics?"
    })

    assert result["intent"] == "policy"
    assert result["response"]["source"] == "policy_kb"
    assert result["response"]["grounded"] is True
    assert result["response"]["llm_mode"] == "MOCK_LLM"

def test_return_risk_route():
    order = {
        "price_inr": 2499,
        "delivery_distance_km": 12.5,
        "customer_tenure_days": 450,
        "delivery_days": 5,
        "discount_pct": 20,
        "num_previous_orders": 8,
        "num_previous_returns": 2,
        "rating_given": 4,
        "payment_method": "COD",
        "product_category": "Footwear",
        "is_weekend_order": 1,
    }

    result = agent_graph.invoke({
        "user_query": "What is the return risk for this order?",
        "order_features": order,
    })

    assert result["intent"] == "return_risk"
    assert result["response"]["source"] == "return_risk_model"
    assert result["response"]["risk_bucket"] == "Medium"
    assert result["response"]["llm_mode"] == "MOCK_LLM"

def test_image_route():
    result = agent_graph.invoke({
        "user_query": "Classify this product image",
        "image_path": "data/sample_images/00_ankle_boot.png",
    })

    assert result["intent"] == "image"
    assert result["response"]["source"] == "image_classifier"
    assert result["response"]["product_class"] == "Ankle boot"
    assert result["response"]["confidence"] > 0.90
    assert result["response"]["llm_mode"] == "MOCK_LLM"

def test_order_context_route():
    result = agent_graph.invoke({
        "user_query": "What about its delivery?",
        "current_order_id": "1523",
    })

    assert result["intent"] == "order_context"
    assert result["current_order_id"] == "1523"
    assert result["response"]["source"] == "conversation_state"
    assert result["response"]["order_id"] == "1523"
    assert "1523" in result["response"]["final_answer"]
    assert result["response"]["llm_mode"] == "MOCK_LLM"