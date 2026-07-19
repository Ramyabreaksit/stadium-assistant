import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from utils import setup_logging

logger = setup_logging(__name__)

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    GENAI_AVAILABLE: bool = True
except ImportError:
    GENAI_AVAILABLE = False
    APIError = Exception


class StadiumAIResponse(BaseModel):
    """Structured Pydantic schema representing the multilingual grounded AI response."""
    detected_language: str = Field(
        description="The full human-readable name of the detected language, e.g., 'English', 'Spanish (Español)', 'Hindi (हिन्दी)', 'French (Français)'"
    )
    detected_language_code: str = Field(
        description="ISO 639-1 language code, e.g., 'en', 'es', 'hi', 'fr', 'ar', 'ja'"
    )
    answer: str = Field(
        description="The final helpful, conversational response answering the user's question grounded in the retrieved FAQ context, written strictly in the detected language."
    )
    is_grounded: bool = Field(
        description="True if the answer directly used facts from the provided context chunks, False if the context did not contain the answer."
    )


def _format_retrieved_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format a list of retrieved ChromaDB dictionary chunks into a structured prompt context string.

    Args:
        retrieved_chunks (List[Dict[str, Any]]): List of retrieved chunk dictionaries.

    Returns:
        str: Formatted context string with numbered source citations.
    """
    if not retrieved_chunks:
        return "No direct factual stadium documents matched this query."
        
    context_text = ""
    for idx, chunk in enumerate(retrieved_chunks, 1):
        stadium = chunk.get("metadata", {}).get("stadium", "General")
        category = chunk.get("metadata", {}).get("category", "info")
        text = chunk.get("text", "")
        context_text += f"\n--- [Source {idx}: {stadium} | {category.upper()}] ---\n{text}\n"
    return context_text


def _format_conversation_history(conversation_history: Optional[List[Dict[str, str]]]) -> str:
    """Format recent turns of conversation history for prompt injection.

    Args:
        conversation_history (Optional[List[Dict[str, str]]]): List of chat turns (`role`, `content`).

    Returns:
        str: Formatted history string or `'None'`.
    """
    if not conversation_history or len(conversation_history) == 0:
        return "None"
        
    history_str = ""
    recent = conversation_history[-3:]
    for turn in recent:
        role = str(turn.get("role", "user")).upper()
        content = str(turn.get("content", ""))
        history_str += f"{role}: {content}\n"
    return history_str.strip()


def _parse_json_response(raw_text: str, default_answer: str = "No answer generated.") -> Dict[str, Any]:
    """Safely parse a JSON model output string into the standardized dictionary structure.

    Args:
        raw_text (str): Raw string output from the Gemini API.
        default_answer (str): Fallback text if the answer field is missing.

    Returns:
        Dict[str, Any]: Standardized response dictionary.
    """
    try:
        data = json.loads(raw_text)
        return {
            "detected_language": str(data.get("detected_language", "Unknown")),
            "detected_language_code": str(data.get("detected_language_code", "en")),
            "answer": str(data.get("answer", default_answer)),
            "is_grounded": bool(data.get("is_grounded", True))
        }
    except Exception as e:
        logger.warning("JSON decoding of raw response failed (%s). Returning raw text.", e)
        return {
            "detected_language": "English",
            "detected_language_code": "en",
            "answer": raw_text or default_answer,
            "is_grounded": True
        }


class GeminiHelper:
    """Helper client handling communication with Google Gemini API for language detection and grounded RAG responses."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        """Initialize the Gemini helper client.

        Args:
            api_key (Optional[str]): Explicit Google Gemini API key. Defaults to environment variables if `None`.
            model_name (str): Target model endpoint for generation (default: `'gemini-2.5-flash'`).
        """
        self.model_name: str = model_name
        self.api_key: Optional[str] = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        
        self.client: Optional[Any] = None
        if GENAI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error("Failed to initialize `google-genai` Client: %s", e)

    def update_api_key(self, api_key: str) -> None:
        """Update the active API key and re-initialize the `google-genai` client.

        Args:
            api_key (str): New Gemini API key string.
        """
        self.api_key = api_key
        if GENAI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error("Failed to re-initialize `google-genai` Client with new key: %s", e)

    def generate_grounded_answer(
        self, 
        user_query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Detect the query language and generate an accurate response grounded strictly in retrieved ChromaDB chunks.

        Args:
            user_query (str): The natural language question submitted by the user.
            retrieved_chunks (List[Dict[str, Any]]): Top-k relevant context chunks retrieved from ChromaDB.
            conversation_history (Optional[List[Dict[str, str]]]): Recent chat turn dictionaries (`role`, `content`).

        Returns:
            Dict[str, Any]: A response dictionary containing `'detected_language'`, `'detected_language_code'`,
            `'answer'`, and `'is_grounded'`.
        """
        if not self.client:
            self.api_key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if self.api_key and GENAI_AVAILABLE:
                try:
                    self.client = genai.Client(api_key=self.api_key)
                except Exception as e:
                    logger.error("Delayed initialization of `google-genai` Client failed: %s", e)
            
            if not self.client:
                return {
                    "detected_language": "English",
                    "detected_language_code": "en",
                    "answer": "⚠️ **API Key Missing or Invalid**: Please enter your valid Google Gemini API Key in the sidebar or set the `GEMINI_API_KEY` environment variable to enable intelligent multilingual RAG responses.",
                    "is_grounded": False
                }

        context_text = _format_retrieved_context(retrieved_chunks)
        history_str = _format_conversation_history(conversation_history)

        system_instruction = (
            "You are 'Stadium Assistant', the official, helpful, and friendly multilingual AI concierge for fans attending the FIFA World Cup 2026.\n"
            "Your critical tasks:\n"
            "1. AUTOMATIC LANGUAGE DETECTION: Analyze the user's question (`user_query`) and identify exactly what language it is written in (e.g. English, Spanish, Hindi, French, Arabic, Portuguese, etc.).\n"
            "2. STRICT GROUNDED RAG ANSWERING: Read the provided `Retrieved Stadium Context Chunks` below. You must answer the user's question directly using ONLY the facts found in those chunks.\n"
            "3. MATCH USER LANGUAGE: You MUST generate your entire `answer` in the EXACT SAME LANGUAGE that the user asked their question in! If the user asks in Hindi, answer in Hindi. If in Spanish, answer in Spanish.\n"
            "4. NO HALLUCINATIONS: If the provided context chunks do not contain the answer to the user's specific question, DO NOT make up gates, transport numbers, or policies. Instead, politely explain in their exact language that the specific detail is not currently in our local stadium FAQ database, and offer to help with other known topics like Gates, Seating tiers, Food/Concessions, Public Transit, Bag Policy, or Restrooms.\n"
            "5. Be welcoming, clear, and professional. Use formatting (bullet points, bold text) where appropriate to make navigation instructions easy to read on mobile devices."
        )

        prompt = (
            f"User Query: {user_query}\n\n"
            f"Recent Conversation History:\n{history_str}\n\n"
            f"Retrieved Stadium Context Chunks:\n{context_text}\n\n"
            "Analyze the query language and provide your grounded response strictly adhering to the JSON schema output format."
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=StadiumAIResponse
                )
            )
            return _parse_json_response(response.text or "{}")
        except APIError as api_err:
            logger.error("Gemini API Error during generation (%s): %s", self.model_name, api_err)
            err_msg = str(api_err)
            if "400" in err_msg or "INVALID_ARGUMENT" in err_msg or "API_KEY_INVALID" in err_msg:
                user_friendly_msg = "⚠️ **Invalid API Key**: The provided Google Gemini API Key is invalid or expired. Please check your key in the sidebar and try again."
            elif "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                user_friendly_msg = "⚠️ **API Quota Exceeded**: You have temporarily exceeded your Gemini API rate limits. Please wait a moment before trying your request again."
            else:
                user_friendly_msg = f"⚠️ **API Communication Error**: Could not complete request due to a Gemini service error (`{err_msg[:120]}...`). Please try again."
            return {
                "detected_language": "English",
                "detected_language_code": "en",
                "answer": user_friendly_msg,
                "is_grounded": False
            }
        except Exception as e:
            logger.warning("Structured schema generation failed (%s). Attempting secondary fallback model...", e)
            try:
                fallback_model = "gemini-2.5-flash" if self.model_name != "gemini-2.5-flash" else "gemini-1.5-flash"
                fallback_prompt = (
                    f"{system_instruction}\n\n"
                    f"User Query: {user_query}\n\n"
                    f"Retrieved Stadium Context Chunks:\n{context_text}\n\n"
                    "Respond strictly in valid JSON format with keys: `detected_language`, `detected_language_code`, `answer`, and `is_grounded` (boolean)."
                )
                response = self.client.models.generate_content(
                    model=fallback_model,
                    contents=fallback_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json"
                    )
                )
                return _parse_json_response(response.text or "{}")
            except Exception as inner_e:
                logger.error("Fallback generation also failed: %s", inner_e)
                return {
                    "detected_language": "English",
                    "detected_language_code": "en",
                    "answer": f"⚠️ **Error generating response**: Unable to complete request right now (`{inner_e}`). Please verify your API key and internet connection.",
                    "is_grounded": False
                }
