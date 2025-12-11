# agents/rag_indexing_agent.py (or merged into your agents.py)

import os
import json
from typing import Dict, List, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# Assuming your state is defined elsewhere
from state import InvoiceProcessingState 
# Assuming you have a logger configured
from logger_config import AUDITOR_LOGGER 

# --- Configuration (Reused from prior steps) ---
VECTOR_DB_PATH = "faiss_invoice_reports"
EMBEDDING_MODEL = "text-embedding-3-small"
REPORTS_FOLDER = "configs/reports" # 👈 NEW: Define the reports directory
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Helper Function for DB Initialization (Reused) ---
def get_vector_store() -> FAISS:
    """Initializes or loads the FAISS vector store with OpenAI Embeddings."""
      
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
       
    if os.path.exists(VECTOR_DB_PATH):
        try:
            vectorstore = FAISS.load_local(
                VECTOR_DB_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            return vectorstore
        except Exception as e:
            # Handle cases where files exist but are corrupted/incomplete
            AUDITOR_LOGGER.error(f"Failed to load existing FAISS index: {e}")
                # Fall through to the creation step if loading fails
            return FAISS.load_local(
                VECTOR_DB_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
    AUDITOR_LOGGER.warning("FAISS index not found. Creating a minimal new index.")
    
    # Create a minimal initial document to bootstrap the FAISS index properly
    dummy_doc = Document(page_content="Initial setup document for FAISS.", metadata={"source": "setup"})   
    # Use the factory method FAISS.from_documents to correctly build the internal structure
    vectorstore = FAISS.from_documents([dummy_doc], embeddings)
    
    return vectorstore

# ----------------------------------------------------

def rag_indexing(state: InvoiceProcessingState) -> Dict:
    """
    Scans the REPORTS_FOLDER for JSON reports, chunks their content, 
    and indexes them into the FAISS vector database.
    """
    AUDITOR_LOGGER.info("Node: rag_indexing | Starting batch indexing from reports folder.")


    all_documents_to_index: List[Document] = []
    report_path = state.get("report_path")
    AUDITOR_LOGGER.info(f"fNode: rag_indexing | report_path data{report_path}")   
    # 1. Load, Process, and Chunk all JSON Reports
    filename = os.path.basename(report_path) 
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
                
        # Assume the report_data is a dictionary with validation results
        # We need to format the dictionary content into a single string for chunking
        AUDITOR_LOGGER.info(f"fNode: rag_indexing | report data{report_data}")       
        invoice_id = report_data.get('indexed_invoice_id', filename.replace('.json', ''))
                
        # Format the full report content into a searchable text block
        report_text = f"--- INVOICE VALIDATION REPORT ({invoice_id}) ---\n"
        report_text += json.dumps(report_data, indent=2, ensure_ascii=False) # Dump the whole structure to a searchable string
                
        # Create LangChain Document
        doc = Document(
            page_content=report_text,
            metadata={
                        "source_file": filename, 
                        "invoice_number": invoice_id
                    }
            )
        all_documents_to_index.append(doc)
       
    except json.JSONDecodeError as e:
        AUDITOR_LOGGER.warning(f"Skipping {filename}: Invalid JSON format. Error: {e}")
                
    except Exception as e:
        AUDITOR_LOGGER.error(f"Error processing file {filename}: {e}")
                

    if not all_documents_to_index:
        AUDITOR_LOGGER.warning("No valid reports found for indexing.")
        return {"indexing_status": "SKIPPED", "errors": state.get('errors', [])}

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    all_chunks = text_splitter.split_documents(all_documents_to_index)
    AUDITOR_LOGGER.info(f"Loaded {len(all_documents_to_index)} documents and split into {len(all_chunks)} chunks.")
    
    # 3. Indexing and Storage
    try:
        vectorstore = get_vector_store()
        
        # Add all new chunks to the vector store (appending to existing index)
        AUDITOR_LOGGER.error(f"all_chunks type: {type(all_chunks)}")
        for i, chunk in enumerate(all_chunks):
    # Check if the content is truly a string
            if not isinstance(chunk.page_content, str):
                AUDITOR_LOGGER.error(f"Chunk {i} has incorrect type: {type(chunk.page_content)}")
        # If it's a tuple or dict, the error will happen.
    
    #  Check for excessive length (which can cause encoding errors or exceed limits)
            if len(chunk.page_content) > 1000: # Check against your CHUNK_SIZE
                AUDITOR_LOGGER.warning(f"Chunk {i} content length is very large: {len(chunk.page_content)}")
        vectorstore.add_documents(all_chunks)
        
        # Save the updated index to disk
        vectorstore.save_local(VECTOR_DB_PATH)
        
        AUDITOR_LOGGER.info(f"Successfully indexed {len(all_chunks)} new chunks.")
        
        return {
            "indexing_status": "SUCCESS",
            "indexed_count": len(all_chunks),
            "errors": state.get('errors', [])
        }
    
    except Exception as e:
        error_msg = f"RAG Indexing/FAISS Error: {e}"
        AUDITOR_LOGGER.error(error_msg)
        return {
            "indexing_status": "FAILURE",
            "indexed_count": 0,
            "errors": state.get('errors', []) + [error_msg]
        }