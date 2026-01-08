from abc import ABC, abstractmethod
import os
import google.genai as genai
from google.genai import types
from openai import OpenAI


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
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        try:
            from openai import AsyncOpenAI
            api_key = os.environ["OPENAI_API_KEY"]
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




class HFProvider(LLMProvider):
    def __init__(self,
                 model: str = "openai/gpt-oss-120b",
                 temperature: float = 0.1,
                 max_tokens: int = 2000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        api_key = os.environ["HF_TOKEN"]
        if not api_key:
            raise ValueError("HF_TOKEN environment variable not set.")
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )
        self.client = client

    async def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
        # print(f"OUT: {completion.choices[0].message.content}")
        return completion.choices[0].message.content




class GeminiProvider:
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.1,
        max_tokens: int = 500
    ):
        self.model_id = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = os.environ["GEMINI_API_KEY"]
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        # The new SDK uses a Client instance
        self.client = genai.Client(api_key=api_key)

    async def generate(self, prompt: str) -> str:
        # The new SDK supports async natively through .aio
        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.95,
                top_k=20,
            ),
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

        # gemini_provider = GeminiProvider()
        # gemini_response = await gemini_provider.generate(prompt)
        # print("\nGemini Response:")
        # print(gemini_response)
        
        
        
        hf_provider = HFProvider()
        hf_response = await hf_provider.generate(prompt)
        print("\nHF Response:")
        print(hf_response)
        
        # mock_provider = MockProvider()
        # mock_response = await mock_provider.generate(prompt)
        # print("\nMock Response:")
        # print(mock_response)

# `asyncio.run(test_providers())` is running the `test_providers()` coroutine function using the
# `asyncio` event loop. This function call is used to execute asynchronous code in Python 3.7 and
# later. It runs the coroutine until it completes and returns the result. In this case, it is testing
# the different providers by generating responses asynchronously.
    asyncio.run(test_providers())
    # hf_provider()
