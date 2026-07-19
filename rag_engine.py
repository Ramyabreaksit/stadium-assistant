import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from knowledge_base.ingest import load_all_faq_documents
from utils import setup_logging

logger = setup_logging(__name__)

COLLECTION_NAME: str = "stadium_faq_gemini_collection"
BATCH_SIZE: int = 50
DEFAULT_TOP_K: int = 4


def _build_chroma_where_clause(
    stadium_filter: Optional[str] = None,
    category_filter: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Helper function to construct a valid ChromaDB where-clause filter dictionary.

    Args:
        stadium_filter (Optional[str]): Venue name to restrict search.
        category_filter (Optional[str]): Topic category filter (`'Gates'`, `'Transport'`, etc.).

    Returns:
        Optional[Dict[str, Any]]: Validated filter dictionary for ChromaDB `where` query argument,
        or `None` if no filters are applied.
    """
    where_filter: Dict[str, Any] = {}
    if stadium_filter and stadium_filter != "All Venues":
        where_filter["stadium"] = stadium_filter
    if category_filter and category_filter != "All Categories":
        where_filter["category"] = category_filter.lower()
        
    if len(where_filter) == 1:
        return where_filter
    elif len(where_filter) > 1:
        return {"$and": [{k: v} for k, v in where_filter.items()]}
    return None


def _format_retrieved_chunk(
    doc_text: str,
    metadata: Dict[str, Any],
    distance: float
) -> Dict[str, Any]:
    """Format and normalize a retrieved raw vector match into a structured result dictionary.

    Args:
        doc_text (str): Document chunk content text.
        metadata (Dict[str, Any]): Associated metadata fields.
        distance (float): Raw L2 or cosine distance score returned by ChromaDB.

    Returns:
        Dict[str, Any]: Dictionary with `text`, `metadata`, normalized `relevance` percentage, and `distance`.
    """
    relevance = max(0.0, min(100.0, round((1.0 - (distance / 2.0)) * 100, 1)))
    return {
        "text": doc_text,
        "metadata": metadata,
        "relevance": relevance,
        "distance": distance
    }


class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """Embedding wrapper around Google Gemini API (`gemini-embedding-001`) via `google-genai` SDK."""

    def __init__(self, model_name: str = "gemini-embedding-001", api_key: Optional[str] = None):
        """Initialize the Gemini embedding function wrapper.

        Args:
            model_name (str): Name of the Gemini embedding model (default: `'gemini-embedding-001'`).
            api_key (Optional[str]): Explicit API key. If `None`, checks `GEMINI_API_KEY` environment variable.
        """
        self.model_name: str = model_name
        self.api_key: Optional[str] = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.fallback_ef = embedding_functions.DefaultEmbeddingFunction()

    @staticmethod
    def name() -> str:
        """Return the unique identifier string of the embedding function for ChromaDB compatibility.

        Returns:
            str: Identifier string (`'gemini-embedding-function'`).
        """
        return "gemini-embedding-function"

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration dictionary for ChromaDB serialization.

        Returns:
            Dict[str, Any]: Configuration dictionary containing model specifications.
        """
        return {"model_name": self.model_name}

    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "GeminiEmbeddingFunction":
        """Reconstruct the embedding function from serialized configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary.

        Returns:
            GeminiEmbeddingFunction: Reconstructed instance.
        """
        return cls(model_name=config.get("model_name", "gemini-embedding-001"))

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of input text strings.

        Args:
            input (List[str]): List of text strings to embed.

        Returns:
            List[List[float]]: List of float embedding vectors (e.g., 3072 dimensions for `gemini-embedding-001`).
        """
        api_key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("No Gemini API key provided for embedding. Using ChromaDB DefaultEmbeddingFunction.")
            return self.fallback_ef(input)
        
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            
            embeddings: List[List[float]] = []
            for text in input:
                resp = client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )
                if resp and resp.embeddings and len(resp.embeddings) > 0:
                    embeddings.append(resp.embeddings[0].values)
                else:
                    embeddings.append(self.fallback_ef([text])[0])
            return embeddings
        except ImportError:
            logger.error("`google-genai` package is not installed. Falling back to default embedding.")
            return self.fallback_ef(input)
        except Exception as e:
            logger.error("Gemini API embedding failed (%s). Falling back to DefaultEmbeddingFunction.", e)
            return self.fallback_ef(input)


class StadiumRAG:
    """Retrieval-Augmented Generation (RAG) engine managing local ChromaDB storage and similarity search."""

    def __init__(self, db_path: str = "chroma_db_data", force_reindex: bool = False):
        """Initialize the persistent ChromaDB client and load/create the vector collection.

        Args:
            db_path (str): Directory path where ChromaDB persists database files.
            force_reindex (bool): Whether to delete existing collection and re-ingest all FAQ files.
        """
        self.db_path: str = db_path
        try:
            os.makedirs(self.db_path, exist_ok=True)
        except OSError as ose:
            logger.error("Failed to create ChromaDB directory at %s: %s", self.db_path, ose)
        
        try:
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
        except Exception as e:
            logger.critical("Failed to initialize persistent ChromaDB client: %s", e)
            raise
        
        self.embedding_function = GeminiEmbeddingFunction()
        
        try:
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={"description": "Multilingual FIFA World Cup 2026 Stadium Knowledge Base (Gemini Embeddings)"}
            )
        except Exception as e:
            logger.error("Failed to get or create collection `%s`: %s", COLLECTION_NAME, e)
            raise
        
        if force_reindex or self.collection.count() == 0:
            self.reindex_knowledge_base()

    def reindex_knowledge_base(self) -> int:
        """Clear existing collection data and re-index all documents from `knowledge_base/data`.

        Returns:
            int: Total number of document chunks successfully indexed into ChromaDB.
        """
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
            
        try:
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={"description": "Multilingual FIFA World Cup 2026 Stadium Knowledge Base (Gemini Embeddings)"}
            )
        except Exception as e:
            logger.error("Failed to recreate ChromaDB collection during re-indexing: %s", e)
            return 0
        
        docs = load_all_faq_documents()
        if not docs:
            logger.info("No documents found to index.")
            return 0
            
        ids = [doc["id"] for doc in docs]
        texts = [doc["text"] for doc in docs]
        metadatas = [doc["metadata"] for doc in docs]
        
        try:
            for i in range(0, len(ids), BATCH_SIZE):
                self.collection.add(
                    ids=ids[i:i+BATCH_SIZE],
                    documents=texts[i:i+BATCH_SIZE],
                    metadatas=metadatas[i:i+BATCH_SIZE]
                )
        except Exception as e:
            logger.error("Failed while adding batch to ChromaDB collection: %s", e)
            
        count = self.collection.count()
        logger.info("Successfully indexed %d items into ChromaDB (%s).", count, COLLECTION_NAME)
        return count

    def get_doc_count(self) -> int:
        """Retrieve the current total document count stored in the active vector collection.

        Returns:
            int: Number of items in ChromaDB collection.
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error("Could not query collection count: %s", e)
            return 0

    def query_stadium_info(
        self, 
        query: str, 
        stadium_filter: Optional[str] = None, 
        category_filter: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """Retrieve top `top_k` relevant FAQ chunks given a user query and optional metadata filters.

        Args:
            query (str): The natural language query or keyword search from the user.
            stadium_filter (Optional[str]): Venue name to restrict search (e.g., `'Estadio Azteca (Mexico City)'`).
            category_filter (Optional[str]): Topic category filter (`'Gates'`, `'Transport'`, etc.).
            top_k (int): Maximum number of document chunks to retrieve (default: 4).

        Returns:
            List[Dict[str, Any]]: List of dictionary results containing `'text'`, `'metadata'`, `'relevance'`, and `'distance'`.
        """
        if not query or not query.strip():
            return []

        chroma_where = _build_chroma_where_clause(stadium_filter, category_filter)
        n_results = min(top_k, max(1, self.collection.count()))
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=chroma_where
            )
        except Exception as e:
            logger.warning("Filter search error (%s). Falling back to unfiltered vector query.", e)
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
            except Exception as inner_e:
                logger.error("ChromaDB query completely failed: %s", inner_e)
                return []
            
        retrieved_chunks: List[Dict[str, Any]] = []
        if results and "documents" in results and results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                doc_text = str(results["documents"][0][i])
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                distance = float(results["distances"][0][i]) if results.get("distances") and results["distances"] else 0.0
                
                retrieved_chunks.append(
                    _format_retrieved_chunk(doc_text, metadata, distance)
                )
                
        return retrieved_chunks


_rag_instance: Optional[StadiumRAG] = None


def get_rag_engine(force_reindex: bool = False) -> StadiumRAG:
    """Retrieve or initialize the global `StadiumRAG` singleton instance.

    Args:
        force_reindex (bool): If `True`, forces re-indexing of the local knowledge base.

    Returns:
        StadiumRAG: Active RAG engine instance.
    """
    global _rag_instance
    if _rag_instance is None or force_reindex:
        _rag_instance = StadiumRAG(force_reindex=force_reindex)
    return _rag_instance
