"""
Groq API Integration module.
Handles LLM calls using official Groq SDK with strict grounding,
low temperature, robust error handling, rate-limit recovery, and retry behavior.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class GroqClientError(Exception):
    """Custom exception wrapper for user-friendly Groq API errors."""
    pass


class GroqService:
    """Service wrapper for Groq LLM inference."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        from config import config
        key = api_key or config.groq_api_key or os.getenv("GROQ_API_KEY", "")
        model = model_name or config.groq_model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.api_key = str(key).strip() if key else ""
        self.model_name = str(model).strip() if model else "openai/gpt-oss-120b"
        self._client = None

    def _get_client(self):
        """Lazy initialization of Groq client."""
        if not self.api_key:
            raise GroqClientError(
                "Groq API Key is not configured. Please enter your Groq API key in the sidebar "
                "or set GROQ_API_KEY in your .env file."
            )

        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise GroqClientError("The 'groq' Python package is not installed.")
            except Exception as e:
                raise GroqClientError(f"Failed to initialize Groq client: {str(e)}")

        return self._client

    def get_available_models(self) -> List[str]:
        """
        Dynamically query active chat completion models supported by this Groq API key.
        """
        try:
            client = self._get_client()
            models_data = client.models.list().data
            # Filter chat models (exclude audio/transcription and guard models)
            chat_models = [
                m.id for m in models_data
                if not m.id.startswith("whisper")
                and "guard" not in m.id
                and "orpheus" not in m.id
            ]
            if chat_models:
                return chat_models
        except Exception as e:
            logger.warning(f"Could not dynamically query Groq models: {e}")
        return ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "allam-2-7b"]

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test API connection with a lightweight ping.
        Returns (success: bool, message: str).
        """
        if not self.api_key:
            return False, "Groq API key is missing."

        try:
            client = self._get_client()
            # Send a minimal 1-token test prompt
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=2,
                temperature=0.0,
            )
            if response and response.choices:
                return True, f"Connected to Groq successfully ({self.model_name})."
            return False, "Received empty response from Groq."
        except Exception as e:
            err_msg = str(e)
            if "authentication" in err_msg.lower() or "invalid api key" in err_msg.lower() or "401" in err_msg:
                return False, "Authentication failed: Invalid Groq API key."
            elif "rate limit" in err_msg.lower() or "429" in err_msg:
                return False, "Rate limit reached on Groq API. Please retry shortly."
            elif "model_not_found" in err_msg.lower() or "does not exist" in err_msg.lower() or "404" in err_msg:
                return False, f"Model '{self.model_name}' is not accessible on your Groq tier. Try 'openai/gpt-oss-120b' or 'openai/gpt-oss-20b'."
            return False, f"Connection error: {err_msg}"

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ) -> str:
        """
        Execute chat completion with robust error handling and friendly error mapping.
        """
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            err_msg = str(e)
            logger.error(f"Groq API completion error: {err_msg}")

            if "401" in err_msg or "invalid api key" in err_msg.lower():
                raise GroqClientError("Invalid Groq API key. Please check your credentials.")
            elif "429" in err_msg or "rate limit" in err_msg.lower():
                raise GroqClientError(
                    "Groq API rate limit exceeded. Please wait a few seconds before trying again."
                )
            elif "model_not_found" in err_msg.lower() or "does not exist" in err_msg.lower() or "404" in err_msg:
                # Try falling back to verified available models on this tier
                fallback_candidates = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
                for fb_model in fallback_candidates:
                    if self.model_name != fb_model:
                        logger.warning(f"Model '{self.model_name}' failed. Automatically falling back to '{fb_model}'...")
                        try:
                            self.model_name = fb_model
                            response = client.chat.completions.create(
                                model=fb_model,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                            if response and response.choices:
                                return response.choices[0].message.content.strip()
                        except Exception as fb_err:
                            logger.warning(f"Fallback to {fb_model} failed: {fb_err}")
                            continue

                raise GroqClientError(f"Model '{self.model_name}' is not accessible on your Groq tier. Please select 'openai/gpt-oss-120b' or 'openai/gpt-oss-20b' in the sidebar.")
            elif "timeout" in err_msg.lower() or "timed out" in err_msg.lower():
                raise GroqClientError("Groq API request timed out. Please verify your internet connection and retry.")
            else:
                raise GroqClientError(f"Groq API error: {err_msg}")

    def generate_suggested_questions(self, sample_texts: List[str]) -> List[str]:
        """
        Generate 3-5 insightful research questions based on document sample passages.
        """
        if not sample_texts:
            return [
                "What are the main research objectives and methodologies?",
                "What are the key findings and empirical results?",
                "What are the limitations and future work discussed?",
            ]

        # Combine snippets up to ~2500 chars
        combined = "\n---\n".join(sample_texts)[:3000]
        prompt = (
            "You are an academic research assistant. Based on the following excerpts from research documents, "
            "generate 4 specific, insightful questions that a researcher would ask to understand or compare these documents.\n"
            "Format your output as exactly 4 bullet points, one question per line, starting with '- '.\n"
            "Do not include introductory or concluding remarks.\n\n"
            f"Excerpts:\n{combined}"
        )

        try:
            res = self.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            questions = []
            for line in res.split("\n"):
                line = line.strip()
                if line.startswith("-") or line.startswith("*") or re.match(r"^\d+\.", line):
                    q = re.sub(r"^[-*\d\.\s]+", "", line).strip()
                    if q and len(q) > 10 and "?" in q:
                        questions.append(q)
            return questions[:4] if questions else [
                "What are the primary findings across the documents?",
                "What methodologies were used by the researchers?",
                "What are the common limitations or challenges identified?",
                "How do the results compare between the papers?",
            ]
        except Exception:
            return [
                "What are the primary findings across the documents?",
                "What methodologies were used by the researchers?",
                "What are the common limitations or challenges identified?",
                "How do the results compare between the papers?",
            ]
