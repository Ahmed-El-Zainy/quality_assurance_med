from abc import ABC, abstractmethod
import os
from openai import OpenAI, AsyncOpenAI


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
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Empty response from OpenAI API")
        return content




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
            max_tokens=self.max_tokens
        )
        content = completion.choices[0].message.content
        if content is None:
            raise ValueError("Empty response from Hugging Face API")
        return content




class GeminiProvider(LLMProvider):
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        self.model_id = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Please set it with: export GEMINI_API_KEY=your-key-here"
            )

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model=self.model_id)
        except ImportError:
            raise ImportError(
                "google-generativeai library not installed. "
                "Install it with: pip install google-generativeai"
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini client: {str(e)}")

    async def generate(self, prompt: str) -> str:
        try:
            response = self.client.generate_content(prompt)
            if response.text is None:
                raise ValueError("Empty response from Gemini API")
            return response.text
        except Exception as e:
            raise ValueError(f"Gemini API error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Gemini API error: {str(e)}")


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
