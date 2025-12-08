# graph_flow/utils.py

from pdfminer.high_level import extract_text_from_file
import pytesseract
from PIL import Image
import os
import docx

def read_file_content(file_path: str) -> str:
    """Reads content from PDF, DOCX, or extracts text from PNG (OCR)."""
    # [Implementation from the previous answer for PDF, DOCX, and PNG handling]
    _, ext = os.path.splitext(file_path.lower())

    if ext == '.pdf':
        return extract_text_from_file(file_path)
    # ... (other file reading logic)
    
    # Placeholder implementation to satisfy the context for this step:
    if ext in ['.pdf', '.docx', '.png']:
         return f"Raw text simulation for {os.path.basename(file_path)}. Invoice Number: INV-2024-456. Date: 2024-11-20. Total: 520.50 EUR. Line Items:\nItem1 | 5 | 100.00 | 500.00\nTAX | 20.50 | 1 | 20.50"
    else:
        raise ValueError(f"Unsupported file format: {ext}")