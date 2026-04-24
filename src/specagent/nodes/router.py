"""
Router node: Determines if a query relates to 3GPP specifications.

Routes queries to either:
    - "retrieve": Query is about 3GPP/telecom, proceed with retrieval
    - "reject": Query is off-topic, return polite rejection

Uses structured output from LLM to get routing decision with reasoning.
"""

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from specagent.llm import get_llm
from specagent.nodes._common import parse_json_object, record_llm_call

if TYPE_CHECKING:
    from specagent.graph.state import GraphState

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    """Structured output for routing decisions."""

    route: Literal["retrieve", "reject"] = Field(
        description="'retrieve' for 3GPP questions, 'reject' for off-topic"
    )
    reasoning: str = Field(description="Brief explanation of the routing decision")


ROUTER_PROMPT = """You are a router for a 3GPP telecom specification RAG assistant.

Task: Route to "retrieve" if the question is likely answerable from 3GPP standards (5G NR, LTE, RAN, Core, NTN/satellite, protocols, parameters, channel models, handover, antenna heights, Doppler shift, propagation, etc.).

Question: {question}

Default to "retrieve" when in doubt, especially for technical/telecom-adjacent questions.

Route to "reject" only if clearly unrelated to telecommunications (cooking, sports, general knowledge, non-telecom programming).

Respond with ONLY a JSON object in this exact format:
{{"route": "retrieve"|"reject", "reasoning": "brief explanation"}}"""


def router_node(state: "GraphState") -> "GraphState":
    """
    Route query to retrieval or rejection.

    Args:
        state: Current graph state containing the user's question

    Returns:
        Updated state with route_decision set to "retrieve" or "reject"
    """
    question = state.get("question", "")

    try:
        llm = get_llm()
        prompt = ROUTER_PROMPT.format(question=question)

        response = llm.invoke(prompt)
        record_llm_call(state, llm, "router")

        decision = RouteDecision(**parse_json_object(response))

        state["route_decision"] = decision.route
        state["route_reasoning"] = decision.reasoning

    except Exception as e:
        # LLM unavailable — default to retrieve so the pipeline can still run
        logger.error("Router LLM error, defaulting to retrieve: %s", e)
        state["route_decision"] = "retrieve"
        state["route_reasoning"] = "LLM unavailable; defaulting to retrieve"
        state["error"] = f"Router LLM error: {e!s}"

    return state
