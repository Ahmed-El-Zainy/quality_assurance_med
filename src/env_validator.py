import os
from typing import Dict, List


class EnvironmentValidator:
    """Validates environment configuration for the API."""
    
    @staticmethod
    def validate_llm_provider() -> bool:
        """
        Validate that the selected LLM provider has required credentials.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        provider = os.getenv("LLM_PROVIDER", "mock").lower()
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI selected but OPENAI_API_KEY not set. "
                    "Set it with: export OPENAI_API_KEY=your-key-here"
                )
        elif provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Gemini selected but GEMINI_API_KEY not set. "
                    "Set it with: export GEMINI_API_KEY=your-key-here"
                )
        elif provider == "hf":
            api_key = os.getenv("HF_TOKEN")
            if not api_key:
                raise ValueError(
                    "Hugging Face selected but HF_TOKEN not set. "
                    "Set it with: export HF_TOKEN=your-token-here"
                )
        elif provider == "mock":
            pass  # Mock provider doesn't need credentials
        else:
            raise ValueError(
                f"Unknown LLM provider: {provider}. "
                f"Must be one of: openai, gemini, hf, mock"
            )
        
        return True
    
    @staticmethod
    def validate_all() -> Dict[str, any]:
        """
        Validate all environment configuration.
        
        Returns:
            Dictionary with validation results
            
        Raises:
            ValueError: If any required configuration is invalid
        """
        results = {
            "llm_provider": os.getenv("LLM_PROVIDER", "mock"),
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", "8000")),
            "valid": False
        }
        
        try:
            EnvironmentValidator.validate_llm_provider()
            results["valid"] = True
            return results
        except ValueError as e:
            results["error"] = str(e)
            raise
