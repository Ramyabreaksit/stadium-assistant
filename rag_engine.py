import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from knowledge_base.ingest import load_all_faq_documents

COLLECTION_NAME: str = "stadium_faq_gemini_collection"


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
            print("[Warning] No Gemini API key provided for embedding. Using ChromaDB DefaultEmbeddingFunction.")
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
            print("[Error] `google-genai` package is not installed. Falling back to default embedding.")
            return self.fallback_ef(input)
        except Exception as e:
            print(f"[Error] Gemini API embedding failed ({e}). Falling back to DefaultEmbeddingFunction.")
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
            print(f"[Error] Failed to create ChromaDB directory at {self.db_path}: {ose}")
        
        # Initialize persistent ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
        except Exception as e:
            print(f"[Critical Error] Failed to initialize persistent ChromaDB client: {e}")
            raise
        
        # Initialize Gemini embedding function (lightweight, no local torch/transformers required)
        self.embedding_function = GeminiEmbeddingFunction()
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={"description": "Multilingual FIFA World Cup 2026 Stadium Knowledge Base (Gemini Embeddings)"}
            )
        except Exception as e:
            print(f"[Error] Failed to get or create collection `{COLLECTION_NAME}`: {e}")
            raise
        
        # Auto-ingest if empty or forced
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
            print(f"[Error] Failed to recreate ChromaDB collection during re-indexing: {e}")
            return 0
        
        docs = load_all_faq_documents()
        if not docs:
            print("[Info] No documents found to index.")
            return 0
            
        ids = [doc["id"] for doc in docs]
        texts = [doc["text"] for doc in docs]
        metadatas = [doc["metadata"] for doc in docs]
        
        # Upsert in batches to ensure smooth ingestion
        batch_size = 50
        try:
            for i in range(0, len(ids), batch_size):
                self.collection.add(
                    ids=ids[i:i+batch_size],
                    documents=texts[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )
        except Exception as e:
            print(f"[Error] Failed while adding batch to ChromaDB collection: {e}")
            
        count = self.collection.count()
        print(f"Successfully indexed {count} items into ChromaDB ({COLLECTION_NAME}).")
        return count

    def get_doc_count(self) -> int:
        """Retrieve the current total document count stored in the active vector collection.

        Returns:
            int: Number of items in ChromaDB collection.
        """
        try:
            return self.collection.count()
        except Exception as e:
            print(f"[Error] Could not query collection count: {e}")
            return 0

    def query_stadium_info(
        self, 
        query: str, 
        stadium_filter: Optional[str] = None, 
        category_filter: Optional[str] = None,
        top_k: int = 4
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

        where_filter: Dict[str, Any] = {}
        if stadium_filter and stadium_filter != "All Venues":
            where_filter["stadium"] = stadium_filter
        if category_filter and category_filter != "All Categories":
            where_filter["category"] = category_filter.lower()
            
        chroma_where: Optional[Dict[str, Any]] = None
        if len(where_filter) == 1:
            chroma_where = where_filter
        elif len(where_filter) > 1:
            chroma_where = {"$and": [{k: v} for k, v in where_filter.items()]}
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, max(1, self.collection.count())),
                where=chroma_where
            )
        except Exception as e:
            print(f"[Warning] Filter search error ({e}). Falling back to unfiltered vector query.")
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(top_k, max(1, self.collection.count()))
                )
            except Exception as inner_e:
                print(f"[Error] ChromaDB query completely failed: {inner_e}")
                return []
            
        retrieved_chunks: List[Dict[str, Any]] = []
        if results and "documents" in results and results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                doc_text = str(results["documents"][0][i])
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                distance = float(results["distances"][0][i]) if results.get("distances") and results["distances"] else 0.0
                
                # Convert L2 distance to relevance percentage approximation (lower distance = higher relevance)
                relevance = max(0.0, min(100.0, round((1.0 - (distance / 2.0)) * 100, 1)))
                
                retrieved_chunks.append({
                    "text": doc_text,
                    "metadata": metadata,
                    "relevance": relevance,
                    "distance": distance
                })
                
        return retrieved_chunks


# Singleton instance accessor for Streamlit caching
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
