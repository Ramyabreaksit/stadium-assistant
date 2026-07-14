import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

class StadiumAIResponse(BaseModel):
    detected_language: str = Field(description="The full human-readable name of the detected language, e.g., 'English', 'Spanish (Español)', 'Hindi (हिन्दी)', 'French (Français)'")
    detected_language_code: str = Field(description="ISO 639-1 language code, e.g., 'en', 'es', 'hi', 'fr', 'ar', 'ja'")
    answer: str = Field(description="The final helpful, conversational response answering the user's question grounded in the retrieved FAQ context, written strictly in the detected language.")
    is_grounded: bool = Field(description="True if the answer directly used facts from the provided context chunks, False if the context did not contain the answer.")

class GeminiHelper:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        
        self.client = None
        if GENAI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize google-genai Client: {e}")

    def update_api_key(self, api_key: str):
        self.api_key = api_key
        if GENAI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to re-initialize google-genai Client: {e}")

    def generate_grounded_answer(
        self, 
        user_query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Detects the query language and generates an accurate answer grounded strictly in the retrieved ChromaDB chunks.
        """
        if not self.client:
            # Check if key is now in env
            self.api_key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if self.api_key and GENAI_AVAILABLE:
                self.client = genai.Client(api_key=self.api_key)
            else:
                return {
                    "detected_language": "English",
                    "detected_language_code": "en",
                    "answer": "⚠️ **API Key Missing**: Please provide your Google Gemini API Key in the sidebar or set the `GEMINI_API_KEY` environment variable to enable intelligent multilingual RAG responses.",
                    "is_grounded": False
                }

        # Format retrieved context
        context_text = ""
        if retrieved_chunks:
            for idx, chunk in enumerate(retrieved_chunks, 1):
                stadium = chunk.get("metadata", {}).get("stadium", "General")
                category = chunk.get("metadata", {}).get("category", "info")
                text = chunk.get("text", "")
                context_text += f"\n--- [Source {idx}: {stadium} | {category.upper()}] ---\n{text}\n"
        else:
            context_text = "No direct factual stadium documents matched this query."

        # Format brief conversation context if any
        history_str = ""
        if conversation_history and len(conversation_history) > 0:
            # Take last 3 turns
            recent = conversation_history[-3:]
            for turn in recent:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                history_str += f"{role.upper()}: {content}\n"

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
            f"Recent Conversation History:\n{history_str if history_str else 'None'}\n\n"
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
            
            # Parse structured JSON response
            raw_text = response.text
            data = json.loads(raw_text)
            return {
                "detected_language": data.get("detected_language", "Unknown"),
                "detected_language_code": data.get("detected_language_code", "en"),
                "answer": data.get("answer", "No answer generated."),
                "is_grounded": data.get("is_grounded", True)
            }
        except Exception as e:
            # Fallback if structured schema generation failed or model choice fell back
            print(f"Error during Gemini response generation ({self.model_name}): {e}")
            # Try plain generation or secondary model (`gemini-2.5-flash` or `gemini-1.5-flash`) if first failed
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
                data = json.loads(response.text)
                return {
                    "detected_language": data.get("detected_language", "English"),
                    "detected_language_code": data.get("detected_language_code", "en"),
                    "answer": data.get("answer", response.text),
                    "is_grounded": data.get("is_grounded", True)
                }
            except Exception as inner_e:
                return {
                    "detected_language": "English",
                    "detected_language_code": "en",
                    "answer": f"⚠️ **Error generating response**: Could not communicate with Gemini API (`{e}`). Please verify your API key and network connection.",
                    "is_grounded": False
                }
