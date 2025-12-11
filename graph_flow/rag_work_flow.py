# graph_flow/workflow.py

from langgraph.graph import StateGraph, END
from graph_flow.state import RAGState
from rag_agents.retrieval_agent import retrieval_agent
from rag_agents.augmentation_agent import augmentation_agent
from rag_agents.generation_agent import generation_agent
from rag_agents.reflection_agent import reflection_agent
# Note: indexing_agent runs once before this graph starts.

MAX_ATTEMPTS = 3 # Maximum times to retry

# --- Conditional Logic ---
def decide_to_retry(state: RAGState) -> str:
    """Checks reflection score to decide if the process needs a retry."""
    score = state.get("reflection_score", {})
    attempt = state.get("attempt", 1)
    
    # Check if Groundedness or Relevance scores are too low (e.g., < 3)
    # RAG Triad scores are 1-3. We retry if any critical score is below max (3).
    is_poor_quality = score.get("groundedness", 3) < 3 or score.get("relevance", 3) < 3
    
    if is_poor_quality and attempt < MAX_ATTEMPTS:
        print(f"Answer needs improvement. Retrying (Attempt {attempt + 1}).")
        return "retry"
    else:
        print("Answer is sufficient or max attempts reached. Finishing.")
        return "finish"

# --- Graph Definition ---
workflow = StateGraph(RAGState)

# 1. Add Nodes
workflow.add_node("retrieve", retrieval_agent)
workflow.add_node("augment", augmentation_agent)
workflow.add_node("generate", generation_agent)
workflow.add_node("reflect", reflection_agent)

# 2. Define the Flow
workflow.set_entry_point("retrieve")

# RAG Chain
workflow.add_edge("retrieve", "augment")
workflow.add_edge("augment", "generate")
workflow.add_edge("generate", "reflect") # After generation, always reflect

# Reflection Loop (Conditional Edge)
workflow.add_conditional_edges(
    "reflect",
    decide_to_retry,
    {
        "retry": "retrieve", # Loop back to retrieval for better chunks
        "finish": END         # End the process
    }
)

rag_app = workflow.compile()