# graph_flow/workflow.py

from langgraph.graph import StateGraph, END
from state import InvoiceProcessingState # Import the state definition
from logger_config import AUDITOR_LOGGER
from agents import (
    ingest_invoice, 
    extract_data, 
    translate_data, 
    validate_invoice_data,
    validate_business_logic,
    reporting,
    rag_indexing,
    decide_to_review,
    decide_final_action,
    decide_to_translate
    # ... import all other agent functions
)

def check_language(state: InvoiceProcessingState) -> str:
    """
    Determines the next step: either skip translation or proceed to it.
    
    Returns:
        'skip_translation' or 'translate'
    """
    lang = state.get("detected_language")
    
    if lang == 'EN':
        # If English, skip the translation agent and jump to validation
        AUDITOR_LOGGER.info("Condition: Language is EN. Skipping translation.")
        return "skip_translation" 
    
    # For any other language (FR, ES, UNKNOWN, etc.), proceed with translation
    AUDITOR_LOGGER.info(f"Condition: Language is {lang}. Proceeding to translation.")
    return "translate"

# 1. Initialize the StateGraph
workflow = StateGraph(InvoiceProcessingState)

# 2. Add the Nodes (Agents)
#workflow.add_node("ingest_invoice", ingest_invoice)
workflow.add_node("extract_data", extract_data)
workflow.add_conditional_edges(
    "extract_data",
    decide_to_translate,
    {
        "translate_data": "translate_data",      # If not 'EN', go to translation
        "validate_invoice_data": "validate_invoice_data", # If 'EN', skip to validation
    }
)
workflow.add_node("translate_data", translate_data)
workflow.add_node("validate_invoice_data", validate_invoice_data)
workflow.add_node("validate_business_logic", validate_business_logic)
workflow.add_node("reporting", reporting)
workflow.add_node("rag_indexing", rag_indexing)

# 3. Set the Entry Point
workflow.set_entry_point("extract_data")

# 4. Define the Edges (Sequential Steps)
#workflow.add_edge("ingest_invoice", "extract_data")
workflow.add_edge("extract_data", "translate_data")
workflow.add_edge("translate_data", "validate_invoice_data")

# 5. Define Conditional Edges (Decision Points)
# Decision after initial (rules.yaml) validation
workflow.add_conditional_edges(
    "validate_invoice_data",
    decide_to_review,
    {
        # If initial rules pass, proceed to ERP check
        "validate_business_logic": "validate_business_logic",
        # If rules fail (missing field, calc error), skip ERP and go straight to report/review
        "reporting": "reporting", 
    }
)

# After Business Validation, always proceed to reporting
workflow.add_edge("validate_business_logic", "reporting")
workflow.add_edge("reporting", "rag_indexing")

# After Reporting, conditionally index or wait for human review
'''workflow.add_conditional_edges(
    "reporting",
    decide_final_action,
    {
        # If Approved, index the report and finish
        "rag_indexing": "rag_indexing",
        # If Manual Review/Reject, the node is an imaginary pause 
        # that exits the automated flow and waits for Streamlit input.
        "human_in_the_loop_wait": END, 
    }
)'''

# 6. Final Edge to End the Automated Flow
workflow.add_edge("rag_indexing", END)

# 7. Compile the Graph
app = workflow.compile()

# --- Example Execution (Simulating an Invoice) ---
# initial_state = app.invoke({"file_path": "/data/incoming/test_invoice.pdf", "file_name": "test_invoice.pdf", "status": "RECEIVED", "errors": []})
# print("\nFinal State:", initial_state)

# You can now import 'app' from other files.