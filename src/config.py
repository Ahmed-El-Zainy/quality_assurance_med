import os
from typing import Literal


class Config:
    # API Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # LLM Provider Settings
    LLM_PROVIDER: Literal["openai", "gemini", "mock", "hf"] = os.getenv("LLM_PROVIDER", "mock")  # type: ignore
    
    # OpenAI Settings
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
    
    # Gemini Settings
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
    GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "2000"))

    # Hugging Face Settings
    HF_MODEL: str = os.getenv("HF_MODEL", "openai/gpt-oss-120b")
    HF_TEMPERATURE: float = float(os.getenv("HF_TEMPERATURE", "0.1"))
    HF_MAX_TOKENS: int = int(os.getenv("HF_MAX_TOKENS", "2000"))

    @classmethod
    def get_llm_provider(cls):
        """Get the configured LLM provider instance."""
        from llm_provider import OpenAIProvider, MockProvider, GeminiProvider, HFProvider
        from env_validator import EnvironmentValidator
        
        # Validate environment before creating provider
        EnvironmentValidator.validate_llm_provider()
        
        provider = cls.LLM_PROVIDER.lower()
        
        if provider == "openai":
            return OpenAIProvider(
                model=cls.OPENAI_MODEL,
                temperature=cls.OPENAI_TEMPERATURE,
                max_tokens=cls.OPENAI_MAX_TOKENS
            )
        elif provider == "gemini":
            return GeminiProvider(
                model=cls.GEMINI_MODEL,
                temperature=cls.GEMINI_TEMPERATURE,
                max_tokens=cls.GEMINI_MAX_TOKENS
            )
        elif provider == "hf":
            return HFProvider(
                model=cls.HF_MODEL,
                temperature=cls.HF_TEMPERATURE,
                max_tokens=cls.HF_MAX_TOKENS
            )
        elif provider == "mock":
            return MockProvider()
        else:
            raise ValueError(
                f"Unknown LLM provider: {cls.LLM_PROVIDER}. "
                f"Must be one of: openai, gemini, hf, mock"
            )


config = Config()

if __name__ == "__main__":
    # Global config instance
    print(f"LLM Provider: {config.LLM_PROVIDER}")
    print(f"Host: {config.HOST}")
    print(f"Port: {config.PORT}")