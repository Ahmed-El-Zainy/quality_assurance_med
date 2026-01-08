"""
LLM Provider Abstraction
Allows swapping between different LLM providers (OpenAI, Anthropic, local models, etc.)
"""
from abc import ABC, abstractmethod
from typing import Optional
import os


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    This allows the system to work with different LLM backends
    without changing the core analysis logic.
    """
    
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
    """
    OpenAI LLM provider using the OpenAI API.
    
    Requires OPENAI_API_KEY environment variable.
    """
    
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


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude LLM provider.
    
    Requires ANTHROPIC_API_KEY environment variable.
    Example of how to add alternative providers.
    """
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            model: Anthropic model name
            temperature: Sampling temperature
            max_tokens: Maximum response length
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        try:
            from anthropic import AsyncAnthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not set. "
                    "Please set it with: export ANTHROPIC_API_KEY=your-key-here"
                )
            self.client = AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "Anthropic library not installed. "
                "Install it with: pip install anthropic"
            )
    
    async def generate(self, prompt: str) -> str:
        """Generate response using Anthropic API."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return response.content[0].text


class MockProvider(LLMProvider):
    """
    Mock provider for testing without API calls.
    
    Returns a fixed response for development/testing.
    """
    
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