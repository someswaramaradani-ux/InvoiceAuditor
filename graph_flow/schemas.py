# graph_flow/schemas.py (Reusing the Pydantic schemas)

from pydantic import BaseModel, Field
from typing import List, Optional

class LineItem(BaseModel):
    item_code: Optional[str] = Field(None, description="The vendor's product or service code.")
    description: str = Field(..., description="The description of the item.")
    qty: float = Field(..., description="The quantity of the item purchased.")
    unit_price: float = Field(..., description="The price per single unit of the item.")
    line_total: float = Field(..., description="The calculated total for this line item (Qty * Unit Price).")

class InvoiceExtractionSchema(BaseModel):
    invoice_number: str
    invoice_date: str
    vendor_name: str
    currency: str
    total_amount: float
    line_items: List[LineItem]

# validation_agent/validation_schema.py

from typing import List, Literal
from pydantic import BaseModel, Field

class Issue(BaseModel):
    code: str = Field(..., description="A short, unique code for the issue, e.g., 'MISSING_FIELD' or 'CALC_MISMATCH'")
    field: str = Field(..., description="The invoice data field related to the issue, e.g., 'total_amount'")
    severity: Literal['High', 'Medium', 'Low'] = Field(..., description="The impact level: High (critical error), Medium (review required), Low (minor issue)")
    message: str = Field(..., description="A detailed explanation of the inconsistency or error.")

class ValidationResults(BaseModel):
    data_valid: bool = Field(..., description="Overall invoice validity status (False if any High severity issue exists).")
    completeness_ok: bool = Field(..., description="True if all required fields are present and valid.")
    calculation_error: bool = Field(..., description="True if a math error (like total or tax) was detected.")

class ValidationReport(BaseModel):
    validation_results: ValidationResults
    issues: List[Issue]