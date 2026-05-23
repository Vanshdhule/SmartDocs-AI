# backend/openai_helper.py
"""
NVIDIA NIM API Helper
Handles key loading, connection testing, and plain text completions.
Points to NVIDIA NIM base URL using the OpenAI-compatible SDK.
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

try:
    from config import CHAT_MODEL, NVIDIA_BASE_URL, NVIDIA_CHAT_API_KEY
except ImportError:
    CHAT_MODEL        = "meta/llama-3.3-70b-instruct"
    NVIDIA_BASE_URL   = "https://integrate.api.nvidia.com/v1"
    NVIDIA_CHAT_API_KEY = None

logger = logging.getLogger("smartdocs.openai_helper")


class OpenAIHelper:

    def __init__(self):
        load_dotenv()
        self.api_key = NVIDIA_CHAT_API_KEY or os.getenv("NVIDIA_CHAT_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_CHAT_API_KEY not found in .env file.")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=NVIDIA_BASE_URL,
        )

    def _handle_error(self, error: Exception) -> None:
        msg = str(error).lower()
        if "api key" in msg or "authentication" in msg or "unauthorized" in msg:
            raise Exception("❌ Invalid NVIDIA API key.")
        if "rate limit" in msg:
            raise Exception("⚠️ Rate limit exceeded.")
        if "quota" in msg or "billing" in msg:
            raise Exception("💳 API quota exceeded.")
        if "connection" in msg or "network" in msg or "timeout" in msg:
            raise Exception("🌐 Network error connecting to NVIDIA NIM.")
        raise Exception(f"Unexpected API error: {error}")

    def test_connection(self) -> bool:
        """Ping the API with a minimal request."""
        try:
            resp = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(resp.choices[0].message.content)
        except Exception as e:
            self._handle_error(e)

    def get_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 200,
    ) -> Optional[str]:
        """Single-turn text completion."""
        try:
            resp = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self._handle_error(e)