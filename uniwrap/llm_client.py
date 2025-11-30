"""Claude API wrapper for making LLM calls."""

import os
import time
from typing import Optional

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


class ClaudeClient:
    """Wrapper for Anthropic Claude API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude client.
        
        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
            
        Raises:
            ValueError: If API key is not provided and not in environment
            ImportError: If anthropic package is not installed
        """
        if Anthropic is None:
            raise ImportError(
                "anthropic package is required. Install it with: pip install anthropic"
            )
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it as an environment variable "
                "or pass it to ClaudeClient(api_key=...)"
            )
        
        self.client = Anthropic(api_key=api_key)
    
    def call_claude(
        self,
        prompt: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_tokens: int = 8192
    ) -> str:
        """Call Claude API with retry logic.

        Args:
            prompt: The prompt to send to Claude
            model: Claude model to use
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            max_tokens: Maximum tokens in response (default 8192 for code generation)

        Returns:
            Raw text response from Claude

        Raises:
            Exception: If all retry attempts fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Extract text from response
                if response.content and len(response.content) > 0:
                    return response.content[0].text
                else:
                    raise ValueError("Empty response from Claude")
                    
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    raise Exception(f"Failed to call Claude after {max_retries} attempts: {last_error}")
        
        raise Exception(f"Unexpected error: {last_error}")

