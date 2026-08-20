from typing import TypedDict, Optional
from part3_support_agent.safety import detect_prompt_injection
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from part3_support_agent.rag import grounded_policy_answer
import re
from part3_support_agent.tools import check_return_risk
from part3_support_agent.tools import (
    check_return_risk,
    classify_product_image,
)
from part3_support_agent.mock_llm import mock_llm_response
from part3_support_agent.prompts import few_shot_intent
class AgentState(TypedDict, total=False):
    user_query: str
    intent: str
    routing_basis: str
    response: dict
    final_response: dict
    order_features: dict
    image_path: str
    current_order_id: Optional[str]
def extract_order_id(user_query: str):
    match = re.search(
        r"\border\s*#?\s*(\d+)\b",
        user_query,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return None

def route_intent(state: AgentState) -> AgentState:
    query = state["user_query"]

    # Capture order ID and preserve it in state
    order_id = extract_order_id(query)

    if order_id:
        state["current_order_id"] = order_id

    # Prompt-injection guardrail
    injection_check = detect_prompt_injection(query)

    if injection_check["blocked"]:
        state["intent"] = "blocked"
        state["response"] = {
            "answer": (
                "I cannot follow instructions that attempt to "
                "override the support assistant's rules."
            ),
            "source": "input_guardrail",
            "confidence": 1.0,
            "blocked": True,
            "reason": injection_check["reason"],
        }
        return state
    # Few-shot routing examples are checked first.
    demonstrated_intent = few_shot_intent(query)

    if demonstrated_intent is not None:
        state["intent"] = demonstrated_intent
        state["routing_basis"] = "few_shot_example"
        return state

    state["routing_basis"] = "deterministic_rules"

    query_lower = query.lower()

    if (
        "return policy" in query_lower
        or "refund" in query_lower
        or "return window" in query_lower
    ):
        intent = "policy"

    elif "return risk" in query_lower or "risk" in query_lower:
        intent = "return_risk"

    elif (
        "category" in query_lower
        or "classify" in query_lower
        or "image" in query_lower
    ):
        intent = "image"

    elif "delivery" in query_lower:
       intent = "order_context"

    else:
        intent = "unknown"

    state["intent"] = intent

    return state

def policy_node(state: AgentState) -> AgentState:
    result = grounded_policy_answer(
        state["user_query"]
    )

    state["response"] = {
        "answer": result["answer"],
        "source": "policy_kb",
        "confidence": result["score"],
        "grounded": result["grounded"],
        "threshold": result["threshold"],
        "sources": result["sources"],
    }

    return state


def return_risk_node(state: AgentState) -> AgentState:
    order_features = state.get("order_features")

    if not order_features:
        state["response"] = {
            "answer": "Order features are required to calculate return risk.",
            "source": "return_risk_tool",
            "confidence": 0.0,
        }
        return state

    result = check_return_risk(order_features)

    state["response"] = {
        "answer": (
            f"Return probability: "
            f"{result['return_probability']:.2%}. "
            f"Risk bucket: {result['risk_bucket']}."
        ),
        "source": "return_risk_tool",
        "confidence": result["return_probability"],
        "threshold": result["threshold"],
        "risk_bucket": result["risk_bucket"],
    }

    return state
def image_node(state: AgentState) -> AgentState:
    image_path = state.get("image_path")

    if not image_path:
        state["response"] = {
            "answer": "An image path is required for product classification.",
            "source": "image_classifier_tool",
            "confidence": 0.0,
        }
        return state

    result = classify_product_image(image_path)

    state["response"] = {
        "answer": (
            f"Predicted product category: "
            f"{result['product_class']}. "
            f"Confidence: {result['confidence']:.2%}."
        ),
        "source": "image_classifier_tool",
        "confidence": result["confidence"],
        "class_id": result["class_id"],
        "product_class": result["product_class"],
    }

    return state


def unknown_node(state: AgentState) -> AgentState:
    return state

def order_context_node(state: AgentState) -> AgentState:
    order_id = state.get("current_order_id")

    if not order_id:
        state["response"] = {
            "answer": (
                "I do not have an order in the current conversation. "
                "Please provide an order ID."
            ),
            "source": "conversation_state",
            "confidence": 1.0,
        }
        return state

    state["response"] = {
        "answer": (
            f"You are asking about order {order_id}. "
            "The order ID was preserved from the current conversation state."
        ),
        "source": "conversation_state",
        "confidence": 1.0,
        "order_id": order_id,
    }

    return state
def response_node(state: AgentState) -> AgentState:
    intent = state.get("intent", "unknown")
    current_response = state.get("response", {})

    if intent == "policy":
        mock_context = {
            "answer": current_response.get(
                "answer",
                "I do not have enough policy evidence to answer."
            )
        }

    elif intent == "return_risk":
        mock_context = {
            "return_probability": current_response.get(
                "confidence",
                0.0
            ),
            "risk_bucket": current_response.get(
                "risk_bucket",
                "Unknown"
            ),
        }

    elif intent == "image":
        mock_context = {
            "product_class": current_response.get(
                "product_class",
                "Unknown"
            ),
            "confidence": current_response.get(
                "confidence",
                0.0
            ),
        }

    elif intent == "order_context":
        mock_context = {
            "order_id": current_response.get(
                "order_id",
                "Unknown"
            ),
            "answer": current_response.get(
                "answer",
                "I do not have enough order context."
            ),
        }

    else:
        mock_context = {}

    llm_result = mock_llm_response(
        intent,
        mock_context
    )

    state["response"] = {
        **current_response,
        "final_answer": llm_result["answer"],
        "llm_mode": "MOCK_LLM",
    }
    if intent in {
        "policy",
        "return_risk",
        "image",
    }:
        state["final_response"] = {
            "answer": llm_result["answer"],
            "source": current_response["source"],
            "confidence": current_response.get(
                "confidence",
                0.0,
            ),
        }

    return state
   
def choose_route(state: AgentState):
    return state["intent"]


graph_builder = StateGraph(AgentState)

graph_builder.add_node("router", route_intent)
graph_builder.add_node("policy", policy_node)
graph_builder.add_node("return_risk", return_risk_node)
graph_builder.add_node("image", image_node)
graph_builder.add_node("order_context", order_context_node)
graph_builder.add_node("unknown", unknown_node)
graph_builder.add_node("response", response_node)

graph_builder.set_entry_point("router")

graph_builder.add_conditional_edges(
    "router",
    choose_route,
    {
        "policy": "policy",
        "return_risk": "return_risk",
        "image": "image",
        "blocked": END,
        "order_context": "order_context",
        "unknown": "unknown",
    },
)

graph_builder.add_edge("policy", "response")
graph_builder.add_edge("order_context", "response")
graph_builder.add_edge("return_risk", "response")
graph_builder.add_edge("image", "response")
graph_builder.add_edge("response", END)

agent_graph = graph_builder.compile()
conversation_memory = InMemorySaver()
stateful_agent_graph = graph_builder.compile(checkpointer=conversation_memory)

def run_agent(
    user_query: str,
    conversation_id: str,
    order_features: Optional[dict] = None,
    image_path: Optional[str] = None,
):
    """
    Run one turn of the support agent.

    LangGraph's InMemorySaver keeps temporary state for turns that use
    the same conversation_id. A different conversation_id starts a
    fresh conversation with no prior order context.
    """
    input_state = {
        "user_query": user_query,
    }

    if order_features is not None:
        input_state["order_features"] = order_features

    if image_path is not None:
        input_state["image_path"] = image_path

    config = {
        "configurable": {
            "thread_id": conversation_id,
        }
    }

    return stateful_agent_graph.invoke(
        input_state,
        config=config,
    )






