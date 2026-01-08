from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List
from datetime import date
import uvicorn

from analyzer import ClinicalNoteAnalyzer
from config import config

app = FastAPI(
    title="Clinical Note QA API",
    description="API for analyzing clinical notes and providing quality assurance feedback",
    version="1.0.0"
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer with configured LLM provider
analyzer = ClinicalNoteAnalyzer(llm_provider=config.get_llm_provider())


# Request models
class NoteMetadata(BaseModel):
    """Metadata for the clinical note"""
    note_type: str = Field(..., description="Type of clinical note (e.g., 'Initial Evaluation', 'Progress Note')")
    date_of_service: date = Field(..., description="Date when service was provided")
    date_of_injury: date = Field(..., description="Date of injury")


class AnalyzeNoteRequest(BaseModel):
    """Request body for note analysis"""
    clinical_note: str = Field(..., description="The clinical note text to analyze")
    metadata: NoteMetadata = Field(..., description="Note metadata")


# Response models
class QAIssue(BaseModel):
    """A single QA issue found in the note"""
    severity: Literal["critical", "major", "minor"] = Field(..., description="Issue severity level")
    issue: str = Field(..., description="Description of why this matters")
    suggested_edit: str = Field(..., description="Minimal suggested edit (1-2 sentences max)")


class AnalyzeNoteResponse(BaseModel):
    """Response containing QA analysis results"""
    score: int = Field(..., ge=0, le=100, description="Overall quality score (0-100)")
    grade: Literal["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D"] = Field(
        ..., description="Letter grade based on score"
    )
    flags: List[QAIssue] = Field(..., min_length=3, max_length=5, description="List of 3-5 QA issues")


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Clinical Note QA API",
        "version": "1.0.0"
    }


@app.post("/analyze-note", response_model=AnalyzeNoteResponse)
async def analyze_note(request: AnalyzeNoteRequest):
    """
    Analyze a clinical note and return quality assessment.
    
    This endpoint:
    1. Validates the input note and metadata
    2. Analyzes the note against QA standards
    3. Returns a score, grade, and actionable improvement suggestions
    
    Args:
        request: The note text and metadata
        
    Returns:
        Analysis results with score, grade, and 3-5 QA issues
        
    Raises:
        HTTPException: If the note cannot be analyzed
    """
    try:
        # Validate input
        if not request.clinical_note.strip():
            raise HTTPException(status_code=400, detail="Clinical note cannot be empty")
        
        # Analyze the note
        result = await analyzer.analyze(
            clinical_note=request.clinical_note,
            note_type=request.metadata.note_type,
            date_of_service=request.metadata.date_of_service.isoformat(),
            date_of_injury=request.metadata.date_of_injury.isoformat()
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing note: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)