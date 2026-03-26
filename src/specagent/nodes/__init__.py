"""
LangGraph nodes for the agentic RAG pipeline.

Each node represents a discrete step in the workflow:
    - router: Determines if query should retrieve documents or be rejected
    - retriever: Fetches relevant chunks from LanceDB vector store
    - grader: Scores relevance of retrieved chunks
    - rewriter: Reformulates query for better retrieval
    - generator: Synthesizes answer from graded chunks
    - hallucination: Verifies answer is grounded in sources

All nodes follow the signature:
    def node_name(state: GraphState) -> GraphState
"""

from specagent.nodes.generator import generator_node
from specagent.nodes.grader import grader_node
from specagent.nodes.hallucination import hallucination_check_node
from specagent.nodes.retriever import retriever_node
from specagent.nodes.rewriter import rewriter_node
from specagent.nodes.router import router_node

__all__ = [
    "generator_node",
    "grader_node",
    "hallucination_check_node",
    "retriever_node",
    "rewriter_node",
    "router_node",
]
