import os
import time
from typing import Tuple, List, Dict, Any, Optional
import streamlit as st
from dotenv import load_dotenv
from utils import setup_logging, verify_requirements_versions

logger = setup_logging(__name__)
load_dotenv()

from rag_engine import get_rag_engine, StadiumRAG
from gemini_helper import GeminiHelper

st.set_page_config(
    page_title="⚽ Stadium Assistant | FIFA World Cup 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css() -> None:
    """Load custom stylesheet (`style.css`) and inject accessible styling into Streamlit DOM."""
    css_path: str = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception as e:
            logger.error("Error loading CSS stylesheet: %s", e)
            st.error(f"Error loading CSS stylesheet: {e}")


def render_grounding_sources(chunks: List[Dict[str, Any]]) -> None:
    """Render expandable UI element displaying retrieved ChromaDB citations and match scores.

    Args:
        chunks (List[Dict[str, Any]]): Retrieved context dictionary results.
    """
    if not chunks or len(chunks) == 0:
        return
    with st.expander(f"🔍 Grounding Sources & ChromaDB Retrieval ({len(chunks)} citations)", expanded=False):
        for idx, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            stadium_name = meta.get("stadium", "General")
            cat = str(meta.get("category", "info")).upper()
            relevance = chunk.get("relevance", 0.0)
            text = chunk.get("text", "")
            
            st.markdown(f"""
            <div class="source-card" role="article" aria-label="Citation Source {idx}">
                <div class="source-header">
                    <span>📌 Source {idx}: {stadium_name} — {cat}</span>
                    <span class="relevance-pill">Match: {relevance}%</span>
                </div>
                <div>{text}</div>
            </div>
            """, unsafe_allow_html=True)


def build_conversation_history(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Format session state messages for Gemini prompt context.

    Args:
        messages (List[Dict[str, Any]]): Raw session state message list.

    Returns:
        List[Dict[str, str]]: Cleaned list of `role` and `content` dictionaries.
    """
    return [
        {"role": str(m["role"]), "content": str(m["content"])}
        for m in messages
    ]


load_css()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "¡Hola! Hello! Bonjour! नमस्ते! ⚽ I am your official **FIFA World Cup 2026 Stadium Assistant**.\n\nYou can ask me **anything in any language** about stadium gates, seating tiers, food options, public transit, restrooms, emergency exits, or bag policies across our tournament venues. How can I help you today?",
            "detected_language": "Multilingual Welcome",
            "detected_language_code": "multi",
            "chunks": [],
            "is_grounded": True
        }
    ]

if "quick_query_trigger" not in st.session_state:
    st.session_state.quick_query_trigger = None


@st.cache_resource(show_spinner=False)
def init_backend() -> Tuple[StadiumRAG, GeminiHelper]:
    """Initialize and cache backend instances (`StadiumRAG` and `GeminiHelper`) and verify requirements.

    Returns:
        Tuple[StadiumRAG, GeminiHelper]: Initialized backend RAG and AI helper instances.
    """
    verify_requirements_versions()
    rag = get_rag_engine(force_reindex=False)
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    gemini = GeminiHelper(api_key=api_key)
    return rag, gemini


rag_engine, gemini_helper = init_backend()

st.markdown("""
<div class="hero-container" role="banner" aria-label="Stadium Assistant Welcome Header">
    <div class="hero-badge" role="status">🏆 FIFA WORLD CUP 2026 — OFFICIAL CONCIERGE</div>
    <div class="hero-title">⚽ Stadium Assistant</div>
    <div class="hero-subtitle">
        Your intelligent, multilingual guide grounded strictly in verified stadium data. Ask anything in English, Spanish, Hindi, French, or any language about gates, seating, food concessions, transit shuttles, restrooms, or security rules.
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Settings & Filters")
    
    current_key: str = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    user_api_key: str = st.text_input(
        "🔑 Gemini API Key",
        value=current_key,
        type="password",
        help="Enter your Google Gemini API Key or set the GEMINI_API_KEY environment variable."
    )
    if user_api_key and user_api_key != current_key:
        os.environ["GEMINI_API_KEY"] = user_api_key
        gemini_helper.update_api_key(user_api_key)
        
    if not (user_api_key or current_key):
        st.warning("⚠️ Please provide a Gemini API Key to enable multilingual generation.")
        
    st.markdown("---")
    st.markdown("### 🗺️ Context Filters")
    
    stadium_filter: str = st.selectbox(
        "🏟️ Filter by Stadium Venue",
        [
            "All Venues",
            "New York / New Jersey Stadium (MetLife)",
            "Estadio Azteca (Mexico City)",
            "Los Angeles Stadium (SoFi)",
            "General Tournament Rules & Policies"
        ],
        index=0,
        help="Filter vector search results to only include facts about a specific venue or rules."
    )
    
    category_filter: str = st.selectbox(
        "📌 Filter by Topic Category",
        [
            "All Categories",
            "Gates",
            "Transport",
            "Seating",
            "Food",
            "Restrooms",
            "Exits",
            "Rules"
        ],
        index=0,
        help="Filter search results by category like navigation, food options, or prohibited items."
    )
    
    st.markdown("---")
    st.markdown("### 📚 Knowledge Base Status")
    doc_count: int = rag_engine.get_doc_count()
    st.markdown(f"""
    <div class="status-box" role="region" aria-label="Knowledge Base Statistics">
        <div class="status-number" aria-live="polite">{doc_count}</div>
        <div class="status-label">Indexed FAQ Documents</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Re-Index Knowledge Base", use_container_width=True, help="Reload all local FAQ files and rebuild the ChromaDB index."):
        with st.spinner("Re-indexing local FAQ files into ChromaDB..."):
            count = rag_engine.reindex_knowledge_base()
            st.success(f"Indexed {count} documents!")
            time.sleep(1)
            st.rerun()
            
    st.markdown("---")
    st.markdown("### 💬 Quick Multilingual Questions")
    st.markdown("<p style='font-size:0.82rem; color:#94a3b8;'>Click any prompt to ask immediately across languages:</p>", unsafe_allow_html=True)
    
    quick_questions: List[str] = [
        "Where is Gate 4 at New York Stadium and what time do gates open?",
        "¿Dónde hay comida vegetariana o halal en el Estadio Azteca?",
        "क्या मैं स्टेडियम के अंदर पानी की बोतल या बैग ले जा सकता हूँ?",
        "Quelles sont les options de transport et navettes pour SoFi Stadium ?",
        "Can I bring my backpack or professional camera inside?"
    ]
    
    for q in quick_questions:
        if st.button(f"👉 {q}", key=f"btn_{q}", help=f"Click to automatically ask: {q}"):
            st.session_state.quick_query_trigger = q

user_input: Optional[str] = st.chat_input("Ask a question in any language (English, Spanish, Hindi, French, Arabic...)...")
if st.session_state.quick_query_trigger:
    user_input = st.session_state.quick_query_trigger
    st.session_state.quick_query_trigger = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            if msg.get("detected_language"):
                st.markdown(f"""<div class="lang-badge" role="status" aria-label="Detected Query Language">🗣️ Query Language: {msg['detected_language']}</div>""", unsafe_allow_html=True)
            st.markdown(msg["content"])
        else:
            if msg.get("detected_language") and msg["detected_language"] != "Multilingual Welcome":
                st.markdown(f"""<div class="lang-badge lang-badge-ai" role="status" aria-label="AI Response Language">🤖 Responding in: {msg['detected_language']}</div>""", unsafe_allow_html=True)
            st.markdown(msg["content"])
            render_grounding_sources(msg.get("chunks", []))

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "detected_language": "Detecting...",
        "chunks": []
    })
    
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing query language & retrieving verified stadium info..."):
            retrieved_chunks = rag_engine.query_stadium_info(
                query=user_input,
                stadium_filter=stadium_filter,
                category_filter=category_filter,
                top_k=4
            )
            
            history = build_conversation_history(st.session_state.messages[:-1])
            response_data = gemini_helper.generate_grounded_answer(
                user_query=user_input,
                retrieved_chunks=retrieved_chunks,
                conversation_history=history
            )
            
            detected_lang: str = response_data.get("detected_language", "English")
            answer_text: str = response_data.get("answer", "")
            is_grounded: bool = response_data.get("is_grounded", True)
            
            st.session_state.messages[-1]["detected_language"] = detected_lang
            
            st.markdown(f"""<div class="lang-badge lang-badge-ai" role="status" aria-label="AI Response Language">🤖 Responding in: {detected_lang}</div>""", unsafe_allow_html=True)
            st.markdown(answer_text)
            render_grounding_sources(retrieved_chunks)
                        
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "detected_language": detected_lang,
                "detected_language_code": response_data.get("detected_language_code", "en"),
                "chunks": retrieved_chunks,
                "is_grounded": is_grounded
            })
            
            time.sleep(0.5)
            st.rerun()
