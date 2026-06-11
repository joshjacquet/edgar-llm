from pydantic import BaseModel

### Goal: Create basemodel for extracted value & description


class ExtractedValue(BaseModel):
    """Represents a single extracted value from the RAG pipeline."""
    query: str
    variable: str
    expected_value: str
    extracted_value: str
    section: str
