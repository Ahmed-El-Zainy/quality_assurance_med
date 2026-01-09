import json
import os
from typing import Dict, Any, Optional
from llm_provider import LLMProvider, HFProvider


class ClinicalNoteAnalyzer:
    """
    Analyzes clinical notes and returns structured QA feedback.
    
    This class coordinates the analysis workflow:
    1. Constructs a detailed prompt with QA rules
    2. Sends the note to an LLM
    3. Parses and validates the response
    4. Ensures output conforms to schema
    """
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm_provider =  HFProvider()
        
    def _build_prompt(
        self,
        clinical_note: str,
        note_type: str,
        date_of_service: str,
        date_of_injury: str
    ) -> str:
        """
        Build the analysis prompt with QA rules and guidelines.
        
        This prompt is carefully structured to:
        - Enforce QA standards
        - Control LLM output format
        - Prevent hallucination
        - Ensure actionable feedback
        """
        return f"""You are a clinical documentation quality assurance specialist. Analyze the following clinical note and provide structured feedback.

CLINICAL NOTE TO ANALYZE:
---
Note Type: {note_type}
Date of Service: {date_of_service}
Date of Injury: {date_of_injury}

{clinical_note}
---

QA STANDARDS (MUST FOLLOW):
1. Do NOT invent facts not present in the note
2. Do NOT rewrite the entire note
3. If information is missing, explicitly state it is missing
4. Distinguish patient-reported history from clinician findings
5. Maintain neutral, defensible language
6. Focus on what would hold up in legal/insurance review

YOUR TASK:
Analyze this note for quality and defensibility. Identify 3-5 specific issues that affect the note's quality.

For each issue:
- Classify severity as: critical, major, or minor
- Explain WHY it matters (legal, clinical, insurance perspective)
- Provide a MINIMAL suggested edit (1-2 sentences maximum)

SEVERITY DEFINITIONS:
- critical: Missing information that could affect care or reimbursement, or language that creates legal risk
- major: Important omissions or unclear documentation that weakens defensibility
- minor: Style or clarity improvements that would enhance professional quality

IMPORTANT CONSTRAINTS:
- Suggested edits should be MINIMAL (1-2 sentences max)
- Do NOT rewrite large sections
- If you suggest adding information, it must be to note that specific information is missing
- Maintain the original clinician's voice and style

OUTPUT FORMAT (JSON only, no other text):
{{
  "score": <integer 0-100>,
  "grade": "<A+, A, A-, B+, B, B-, C+, C, C-, or D>",
  "flags": [
    {{
      "severity": "<critical|major|minor>",
      "issue": "<explanation of why this matters>",
      "suggested_edit": "<minimal 1-2 sentence suggestion>"
    }}
  ]
}}

GRADING SCALE:
- 95-100: A+
- 90-94: A
- 85-89: A-
- 80-84: B+
- 75-79: B
- 70-74: B-
- 65-69: C+
- 60-64: C
- 55-59: C-
- 0-54: D

Return ONLY valid JSON. No preamble, no explanation, just the JSON object."""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse and validate the LLM response.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed and validated dictionary
            
        Raises:
            ValueError: If response cannot be parsed or is invalid
        """
        # Try to extract JSON from the response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            # Find the start and end of JSON
            lines = response.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_json = not in_json
                    continue
                if in_json or (line.strip().startswith("{") and not in_json):
                    json_lines.append(line)
                    in_json = True
            response = "\n".join(json_lines)
        
        # Parse JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}")
        
        # Validate structure
        required_keys = {"score", "grade", "flags"}
        if not all(key in data for key in required_keys):
            raise ValueError(f"Response missing required keys. Expected: {required_keys}")
        
        # Validate score
        if not isinstance(data["score"], int) or not 0 <= data["score"] <= 100:
            raise ValueError("Score must be an integer between 0 and 100")
        
        # Validate grade
        valid_grades = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D"]
        if data["grade"] not in valid_grades:
            raise ValueError(f"Grade must be one of: {valid_grades}")
        
        # Validate flags
        if not isinstance(data["flags"], list) or not 3 <= len(data["flags"]) <= 5:
            raise ValueError("Flags must be a list with 3-5 items")
        
        for flag in data["flags"]:
            if not all(key in flag for key in ["severity", "issue", "suggested_edit"]):
                raise ValueError("Each flag must have severity, issue, and suggested_edit")
            
            if flag["severity"] not in ["critical", "major", "minor"]:
                raise ValueError("Severity must be critical, major, or minor")
        
        return data
    
    async def analyze(
        self,
        clinical_note: str,
        note_type: str,
        date_of_service: str,
        date_of_injury: str
    ) -> Dict[str, Any]:
        """
        Analyze a clinical note and return QA results.
        
        Args:
            clinical_note: The note text
            note_type: Type of note
            date_of_service: Service date (ISO format)
            date_of_injury: Injury date (ISO format)
            
        Returns:
            Dictionary with score, grade, and flags
            
        Raises:
            ValueError: If analysis fails or response is invalid
        """
        # Build the prompt
        prompt = self._build_prompt(
            clinical_note=clinical_note,
            note_type=note_type,
            date_of_service=date_of_service,
            date_of_injury=date_of_injury
        )
        
        # Get LLM response
        response = await self.llm_provider.generate(prompt)
        
        # Parse and validate
        result = self._parse_response(response)
        
        return result
    
    
    
if __name__ == "__main__":
    import asyncio

    async def test():
        analyzer = ClinicalNoteAnalyzer()
        sample_note = """Patient presents with complaints of lower back pain for the past 2 weeks. No history of trauma. Physical examination reveals tenderness over the lumbar region. Range of motion is limited due to pain. No neurological deficits noted. Plan includes NSAIDs and physical therapy."""
        result = await analyzer.analyze(
            clinical_note=sample_note,
            note_type="Initial Evaluation",
            date_of_service="2024-01-15",
            date_of_injury="2024-01-14"
        )
        print(json.dumps(result, indent=2))

    asyncio.run(test())
