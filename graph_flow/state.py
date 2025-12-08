from typing import TypedDict, List, Literal, Dict, Optional
from schemas import ValidationReport
from langgraph.graph import StateGraph, END, START

# Define the structure of the data passed through the graph
class InvoiceProcessingState(TypedDict):
    """
    Represents the state of the invoice as it moves through the pipeline.
    """
    # Ingestion & File Info
    file_path: str
    file_name: str
    status: Literal["RECEIVED", "PROCESSING", "REVIEW", "REJECTED", "APPROVED", "INDEXED"]
    
    # Extraction & Raw Data
    detected_language: Optional[str]
    extracted_data: str
    
    # Translation Data
    source_language: str
    translated_data: Dict
    translation_confidence: float
    
    # Validation & Auditing
    validation_results: Dict # Stores results from both Data and Business agents
    validation_issues: List[str]
    business_validation_results: Dict
    business_validation_issues: Dict
    audit_score: float # Overall compliance score
    
    # Reporting & RAG
    report_path: str
    recommendation: Literal["Approve", "Manual Review", "Reject"]
    human_feedback: str # Captured by Streamlit
    errors: List[str]