import os
from typing import Literal


class Config:
    # API Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # LLM Provider Settings
    LLM_PROVIDER: Literal["openai", "gemini", "mock", "hf"] = os.getenv("LLM_PROVIDER", "hf")  # type: ignore
    
    # OpenAI Settings
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
    
    # Gemini Settings
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
    GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "2000"))

    # Hugging Face Settings
    HF_MODEL: str = os.getenv("HF_MODEL", "openai/gpt-oss-120b")
    HF_TEMPERATURE: float = float(os.getenv("HF_TEMPERATURE", "0.1"))
    HF_MAX_TOKENS: int = int(os.getenv("HF_MAX_TOKENS", "2000"))

    @classmethod
    def get_llm_provider(cls):
        from llm_provider import OpenAIProvider, MockProvider, GeminiProvider, HFProvider
        if cls.LLM_PROVIDER == "openai":
            return OpenAIProvider(
                model=cls.OPENAI_MODEL,
                temperature=cls.OPENAI_TEMPERATURE,
                max_tokens=cls.OPENAI_MAX_TOKENS
            )
            
        
        elif cls.LLM_PROVIDER == "gemini":
            return GeminiProvider(
                model=cls.GEMINI_MODEL,
                temperature=cls.GEMINI_TEMPERATURE,
                max_tokens=cls.GEMINI_MAX_TOKENS
            )
            
        
        elif cls.LLM_PROVIDER == "hf":
            return HFProvider(
                model=cls.HF_MODEL,
                temperature=cls.HF_TEMPERATURE,
                max_tokens=cls.HF_MAX_TOKENS
            )
            
        elif cls.LLM_PROVIDER == "mock":
            return MockProvider()
        
        else:
            raise ValueError(
                f"Unknown LLM provider: {cls.LLM_PROVIDER}. "
                f"Must be one of: openai, gemini, mock"
            )


config = Config()
if __name__ == "__main__":
    # Global config instance
    config = Config()
    print(f"LLM Provider: {config.LLM_PROVIDER}")