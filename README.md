# Clinical Note QA API

An AI-powered Quality Assurance engine that analyzes clinical notes and provides structured feedback on quality and defensibility.

## Overview

This API accepts clinical notes and returns:

* **Quality score** (0-100)
* **Letter grade** (A+ to D)
* **3-5 actionable QA issues** with severity, explanation, and minimal suggested edits

The system is designed to help improve clinical documentation quality while maintaining neutral, defensible language suitable for legal and insurance review.

## Quick Start

### Prerequisites

* Python 3.9 or higher
* OpenAI API key (or configure alternative LLM provider)

### Installation

1. **Clone or extract the repository**

   ```bash
   cd clinical-qa-api
   ```
2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Set up your API key**

   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

   On Windows:

   ```cmd
   set OPENAI_API_KEY=your-openai-api-key-here
   ```

### Running the API

Start the server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`

You can view the interactive API documentation at:

* Swagger UI: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`

## Usage

### Example Request

```bash
curl -X POST "http://localhost:8000/analyze-note" \
  -H "Content-Type: application/json" \
  -d '{
    "clinical_note": "Patient presents with lower back pain. Pain started after lifting heavy box at work. Patient rates pain 7/10. Prescribed ibuprofen.",
    "metadata": {
      "note_type": "Initial Evaluation",
      "date_of_service": "2024-01-15",
      "date_of_injury": "2024-01-14"
    }
  }'
```

### Example Response

```json
{
  "score": 68,
  "grade": "C+",
  "flags": [
    {
      "severity": "critical",
      "issue": "Missing objective findings from physical examination. Without documented clinical findings, the note cannot support medical necessity for treatment or justify the prescribed intervention.",
      "suggested_edit": "Add objective examination findings: 'Physical exam revealed tenderness to palpation over L4-L5, positive straight leg raise at 45 degrees, and reduced range of motion with forward flexion limited to 30 degrees.'"
    },
    {
      "severity": "major",
      "issue": "Patient-reported history is not clearly distinguished from clinician observations, creating potential ambiguity in documentation that could be challenged in legal or insurance review.",
      "suggested_edit": "Separate subjective and objective sections. Start with 'SUBJECTIVE: Patient reports...' followed by 'OBJECTIVE: Physical examination reveals...'"
    },
    {
      "severity": "major",
      "issue": "No documentation establishing causation between the work injury and current symptoms, which is critical for workers' compensation claims and insurance authorization.",
      "suggested_edit": "Add: 'Current lower back pain is consistent with and causally related to the 1/14/24 work injury based on mechanism of injury, symptom onset timing, and anatomical distribution of complaints.'"
    }
  ]
}
```

## Architecture

### Components

```
clinical-qa-api/
├── main.py              # FastAPI application and endpoints
├── analyzer.py          # Core analysis logic and prompt engineering
├── llm_provider.py      # LLM abstraction layer
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### How It Works

1. **API Layer** (`main.py`)
   * FastAPI endpoint receives note + metadata
   * Validates input using Pydantic models
   * Returns structured JSON response
2. **Analyzer** (`analyzer.py`)
   * Constructs detailed QA prompt with rules
   * Enforces non-negotiable standards:
     * No fact invention
     * No complete rewrites
     * Explicit acknowledgment of missing info
     * Separation of patient history vs. clinician findings
     * Neutral, defensible language
   * Parses and validates LLM response
   * Ensures output conforms to expected schema
3. **LLM Provider** (`llm_provider.py`)
   * Abstract interface for LLM backends
   * Default: OpenAI (gpt-4o-mini for cost efficiency)
   * Easily swappable to other providers (Anthropic, local models, etc.)

### Prompt Engineering Strategy

The prompt is carefully designed to:

1. **Control Output Format** : Strict JSON schema with exact field names
2. **Enforce QA Standards** : Explicit rules embedded in prompt
3. **Prevent Hallucination** : "Do not invent facts" repeated multiple times
4. **Ensure Actionability** : Requires specific, minimal edits (1-2 sentences)
5. **Maintain Consistency** : Low temperature (0.1) for reproducible results

### LLM Control Techniques

1. **Structured Output** : Uses JSON mode and explicit schema
2. **Few-shot Learning** : Severity definitions provided inline
3. **Constraint Enforcement** : Repeated reminders about minimal edits
4. **Response Validation** : Multi-layer parsing with error handling
5. **System Prompting** : Separate system message to reinforce role

## Design Decisions & Tradeoffs

### 1. **Model Selection: GPT-4o-mini**

* **Why** : Balance of quality and cost
* **Tradeoff** : More sophisticated models (GPT-4) would provide better analysis but cost 15-30x more
* **Alternative** : Can easily swap to Claude or other models via `llm_provider.py`

### 2. **Low Temperature (0.1)**

* **Why** : Consistency and reliability over creativity
* **Tradeoff** : Less variation in responses, but more predictable output
* **Reasoning** : QA needs to be consistent and reproducible

### 3. **3-5 Flags Required**

* **Why** : Balances thoroughness with actionability
* **Tradeoff** : Might force minor issues on good notes, or miss issues on very poor notes
* **Reasoning** : Provides consistent user experience and prevents overwhelming feedback

### 4. **Single-Stage Analysis**

* **Why** : Simplicity and speed
* **Tradeoff** : Multi-stage prompting (first identify issues, then generate edits) might improve quality
* **Reasoning** : Proof of concept prioritizes simplicity; can refine later

### 5. **No Caching/Retry Logic**

* **Why** : Keeps code simple for initial implementation
* **Tradeoff** : API failures or rate limits cause immediate errors
* **Future Enhancement** : Add exponential backoff and response caching

### 6. **Synchronous-Style Async**

* **Why** : FastAPI best practices with async/await
* **Tradeoff** : Slightly more complex than pure sync, but better for production scaling
* **Reasoning** : Prepares codebase for concurrent request handling

### 7. **Minimal Error Context**

* **Why** : Don't expose LLM internals or prompts to API consumers
* **Tradeoff** : Harder to debug without logs
* **Future Enhancement** : Add structured logging (not implemented to keep scope narrow)

## Swapping LLM Providers

The system is designed for easy LLM swapping:

### Using Anthropic Claude

```python
# In main.py, modify the analyzer initialization:
from llm_provider import AnthropicProvider

analyzer = ClinicalNoteAnalyzer(
    llm_provider=AnthropicProvider(model="claude-3-5-sonnet-20241022")
)
```

### Using a Custom Provider

Implement the `LLMProvider` interface:

```python
from llm_provider import LLMProvider

class MyCustomProvider(LLMProvider):
    async def generate(self, prompt: str) -> str:
        # Your implementation here
        return response_text
```

## Testing

### Manual Testing

Use the interactive docs at `http://localhost:8000/docs` to test the API.

### Sample Test Cases

**Good Note** (should score 85+):

* Complete subjective and objective sections
* Clear physical exam findings
* Documented causation to injury
* Proper separation of patient reports vs. clinical findings

**Poor Note** (should score <70):

* Missing physical exam
* No causation documented
* Vague or invented details
* Mixed subjective/objective information

### Automated Testing

Basic test structure (not implemented but recommended):

```python
import pytest
from analyzer import ClinicalNoteAnalyzer
from llm_provider import MockProvider

@pytest.mark.asyncio
async def test_basic_analysis():
    analyzer = ClinicalNoteAnalyzer(llm_provider=MockProvider())
    result = await analyzer.analyze(
        clinical_note="Test note",
        note_type="Progress Note",
        date_of_service="2024-01-15",
        date_of_injury="2024-01-10"
    )
    assert 0 <= result["score"] <= 100
    assert result["grade"] in ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D"]
    assert 3 <= len(result["flags"]) <= 5
```

## Production Considerations

**NOT implemented** (intentionally out of scope):

* ❌ Authentication/authorization
* ❌ Rate limiting
* ❌ HIPAA compliance infrastructure
* ❌ Database/persistence
* ❌ Comprehensive logging
* ❌ Monitoring/observability
* ❌ EMR integration
* ❌ Multi-tenancy

**Would be needed** for production:

* ✅ Authentication (e.g., API keys, OAuth)
* ✅ HIPAA BAA with LLM provider
* ✅ Data encryption at rest and in transit
* ✅ Audit logging of all analyzed notes
* ✅ Rate limiting per user/organization
* ✅ Response caching for identical notes
* ✅ Retry logic with exponential backoff
* ✅ Health checks and monitoring
* ✅ Database for storing analysis history
* ✅ Batch processing for multiple notes

## API Reference

### `POST /analyze-note`

Analyze a clinical note and return QA feedback.

**Request Body:**

```json
{
  "clinical_note": "string",
  "metadata": {
    "note_type": "string",
    "date_of_service": "YYYY-MM-DD",
    "date_of_injury": "YYYY-MM-DD"
  }
}
```

**Response:**

```json
{
  "score": 0-100,
  "grade": "A+ to D",
  "flags": [
    {
      "severity": "critical|major|minor",
      "issue": "string",
      "suggested_edit": "string (1-2 sentences)"
    }
  ]
}
```

**Status Codes:**

* `200`: Success
* `400`: Invalid input
* `500`: Server error

## License & Ownership

All submitted code becomes the property of FirstImpact Med.

## Questions or Issues?

This is a proof-of-concept implementation demonstrating:

- Following detailed specs
- Controlling LLM output
- Clean, readable backend code
- Clear technical decision-making
