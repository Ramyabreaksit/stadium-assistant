import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from knowledge_base.ingest import load_all_faq_documents

COLLECTION_NAME = "stadium_faq_collection"

class MultilingualEmbeddingFunction(chromadb.EmbeddingFunction):
    """Custom embedding wrapper around SentenceTransformers to ensure cross-lingual matching."""
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load {model_name} via sentence_transformers ({e}). Falling back to ChromaDB default embedding.")
            self._model_loaded = False
            self.fallback_ef = embedding_functions.DefaultEmbeddingFunction()

    def __call__(self, input: List[str]) -> List[List[float]]:
        if self._model_loaded:
            embeddings = self.model.encode(input, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            return self.fallback_ef(input)

class StadiumRAG:
    def __init__(self, db_path: str = "chroma_db_data", force_reindex: bool = False):
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize multilingual embedding function for accurate cross-lingual retrieval
        self.embedding_function = MultilingualEmbeddingFunction()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"description": "Multilingual FIFA World Cup 2026 Stadium Knowledge Base"}
        )
        
        # Auto-ingest if empty or forced
        if force_reindex or self.collection.count() == 0:
            self.reindex_knowledge_base()

    def reindex_knowledge_base(self) -> int:
        """Clear and reload all documents from knowledge_base/data into ChromaDB."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
            
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"description": "Multilingual FIFA World Cup 2026 Stadium Knowledge Base"}
        )
        
        docs = load_all_faq_documents()
        if not docs:
            print("No documents found to index.")
            return 0
            
        ids = [doc["id"] for doc in docs]
        texts = [doc["text"] for doc in docs]
        metadatas = [doc["metadata"] for doc in docs]
        
        # Upsert in batches to ensure smooth ingestion
        batch_size = 50
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
            
        count = self.collection.count()
        print(f"Successfully indexed {count} items into ChromaDB ({COLLECTION_NAME}).")
        return count

    def get_doc_count(self) -> int:
        return self.collection.count()

    def query_stadium_info(
        self, 
        query: str, 
        stadium_filter: Optional[str] = None, 
        category_filter: Optional[str] = None,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant FAQ chunks given a user query, with optional metadata filtering."""
        where_filter = {}
        if stadium_filter and stadium_filter != "All Venues":
            # Exact or partial stadium match if specified
            where_filter["stadium"] = stadium_filter
        if category_filter and category_filter != "All Categories":
            where_filter["category"] = category_filter.lower()
            
        # Chroma where format: if multiple filters, wrap in $and
        chroma_where = None
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
            # If where filter failed or returned no results, fallback without strict filter
            print(f"Filter search error or empty ({e}), falling back to unfiltered vector query.")
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, max(1, self.collection.count()))
            )
            
        retrieved_chunks = []
        if results and "documents" in results and results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                doc_text = results["documents"][0][i]
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
                
                # Convert L2 distance or cosine to relevance score percentage approximation
                # Lower distance = higher relevance
                relevance = max(0.0, min(100.0, round((1.0 - (distance / 2.0)) * 100, 1)))
                
                retrieved_chunks.append({
                    "text": doc_text,
                    "metadata": metadata,
                    "relevance": relevance,
                    "distance": distance
                })
                
        return retrieved_chunks

# Singleton instance accessor for Streamlit caching
_rag_instance = None

def get_rag_engine(force_reindex: bool = False) -> StadiumRAG:
    global _rag_instance
    if _rag_instance is None or force_reindex:
        _rag_instance = StadiumRAG(force_reindex=force_reindex)
    return _rag_instance
