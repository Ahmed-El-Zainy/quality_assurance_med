from abc import ABC, abstractmethod
from typing import Optional
import os
import google.generativeai as genai


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response as a string
        """
        pass


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            model: OpenAI model name (default: gpt-4o-mini for cost efficiency)
            temperature: Sampling temperature (low for consistency)
            max_tokens: Maximum response length
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Lazy import to allow running without openai installed
        try:
            from openai import AsyncOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. "
                    "Please set it with: export OPENAI_API_KEY=your-key-here"
                )
            self.client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. "
                "Install it with: pip install openai"
            )
    
    async def generate(self, prompt: str) -> str:
        """
        Generate response using OpenAI API.
        
        Uses a low temperature for consistent, predictable outputs.
        Includes response_format to encourage JSON output.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a clinical documentation QA specialist. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"}
        )
        
        return response.choices[0].message.content


class GeminiProvider: # Assuming LLMProvider is defined elsewhere
    def __init__(
        self,
        model: str = "gemini-2.0-flash", # Updated to a current version
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable not set. "
                    "Please set it with: export GEMINI_API_KEY=your-key-here"
                )
            
            # Configure the library
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            
        except ImportError:
            raise ImportError(
                "Google Generative AI library not installed. "
                "Install it with: pip install google-generativeai"
            )

    async def generate(self, prompt: str) -> str:
        # Note: generate_content_async is used for async operations
        generation_config = {
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 20,
            "max_output_tokens": self.max_tokens,
        }
        
        response = await self.model.generate_content_async(
            prompt,
            generation_config=generation_config
        )
        
        return response.text



class MockProvider(LLMProvider):
    async def generate(self, prompt: str) -> str:
        """Return a mock response."""
        return """{
  "score": 75,
  "grade": "B",
  "flags": [
    {
      "severity": "critical",
      "issue": "Missing objective findings from physical examination. Without documented clinical findings, the note cannot support medical necessity for treatment.",
      "suggested_edit": "Add: 'Physical examination revealed [specific findings]. Range of motion testing showed [measurements].'"
    },
    {
      "severity": "major",
      "issue": "Patient-reported symptoms are not clearly distinguished from clinician observations, which could create ambiguity in legal review.",
      "suggested_edit": "Rephrase subjective complaints to: 'Patient reports...' and document objective findings separately under 'Physical Exam:'"
    },
    {
      "severity": "major",
      "issue": "No documentation of how current presentation relates to date of injury, weakening causation argument for workers' comp or insurance.",
      "suggested_edit": "Add statement: 'Current symptoms are consistent with and causally related to the [date] injury based on [reasoning].'"
    }
  ]
}"""





if __name__ == "__main__":    # Simple test of providers
    import asyncio

    async def test_providers():
        prompt = "Generate a clinical note QA analysis for testing."
        
        # openai_provider = OpenAIProvider()
        # openai_response = await openai_provider.generate(prompt)
        # print("OpenAI Response:")
        # print(openai_response)
        
        gemini_provider = GeminiProvider()
        gemini_response = await gemini_provider.generate(prompt)
        print("\nGemini Response:")
        print(gemini_response)
        
        mock_provider = MockProvider()
        mock_response = await mock_provider.generate(prompt)
        print("\nMock Response:")
        print(mock_response)
    
    asyncio.run(test_providers())