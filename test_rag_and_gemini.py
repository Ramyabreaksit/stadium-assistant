import unittest
import os
import shutil
from typing import List, Dict, Any
from knowledge_base.ingest import load_all_faq_documents
from rag_engine import StadiumRAG
from gemini_helper import GeminiHelper


class TestStadiumAssistant(unittest.TestCase):
    """Unit test suite verifying knowledge base ingestion, ChromaDB vector retrieval, and Gemini AI helper behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up temporary test directory and initialize clean environment."""
        cls.test_db_path: str = "test_chroma_db"
        if os.path.exists(cls.test_db_path):
            shutil.rmtree(cls.test_db_path)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up and remove temporary test ChromaDB folder."""
        if os.path.exists(cls.test_db_path):
            try:
                shutil.rmtree(cls.test_db_path)
            except Exception:
                pass

    def test_01_load_faq_documents(self) -> None:
        """Verify that FAQ documents (.json, .md, .txt) are correctly loaded and structured."""
        docs: List[Dict[str, Any]] = load_all_faq_documents()
        self.assertGreater(len(docs), 0, "Should load at least some FAQ chunks from data directory")
        sample = docs[0]
        self.assertIn("id", sample)
        self.assertIn("text", sample)
        self.assertIn("metadata", sample)
        self.assertIn("stadium", sample["metadata"])

    def test_02_rag_indexing_and_retrieval(self) -> None:
        """Verify ChromaDB indexing and basic multilingual semantic retrieval across venues."""
        rag = StadiumRAG(db_path=self.test_db_path, force_reindex=True)
        count = rag.get_doc_count()
        self.assertGreater(count, 0, "ChromaDB should have indexed documents")
        
        # Test English query about Gates at MetLife
        results_en = rag.query_stadium_info("Where is Gate 4 at New York / New Jersey Stadium?")
        self.assertGreater(len(results_en), 0)
        self.assertIn("MetLife", results_en[0]["metadata"]["stadium"])
        
        # Test Spanish query about Halal/Vegetarian food at Azteca
        results_es = rag.query_stadium_info("¿Dónde hay comida vegetariana en el Estadio Azteca?")
        self.assertGreater(len(results_es), 0)
        
        # Test filter by category
        results_rules = rag.query_stadium_info("What items are prohibited?", category_filter="Rules")
        self.assertGreater(len(results_rules), 0)
        self.assertEqual(results_rules[0]["metadata"]["category"], "rules")

    def test_03_query_stadium_info_advanced(self) -> None:
        """Test advanced behavior of `query_stadium_info` including top_k limits, empty queries, combined filters, and relevance bounds."""
        rag = StadiumRAG(db_path=self.test_db_path, force_reindex=False)
        
        # 1. Test top_k limit behavior
        results_top2 = rag.query_stadium_info("gates and transport", top_k=2)
        self.assertLessEqual(len(results_top2), 2, "Should respect top_k limit of 2")
        
        # 2. Test empty or whitespace query handling
        results_empty = rag.query_stadium_info("")
        self.assertEqual(len(results_empty), 0, "Empty query should return empty list without error")
        results_spaces = rag.query_stadium_info("   ")
        self.assertEqual(len(results_spaces), 0, "Whitespace query should return empty list")
        
        # 3. Test combined stadium and category filters
        results_combined = rag.query_stadium_info(
            query="shuttles and parking options",
            stadium_filter="Los Angeles Stadium (SoFi)",
            category_filter="Transport"
        )
        self.assertGreater(len(results_combined), 0)
        for chunk in results_combined:
            meta = chunk.get("metadata", {})
            self.assertIn("SoFi", meta.get("stadium", ""))
            self.assertEqual(meta.get("category", ""), "transport")
            
        # 4. Test relevance scores are properly normalized percentages between 0.0 and 100.0
        for chunk in results_combined:
            relevance = chunk.get("relevance", -1.0)
            self.assertGreaterEqual(relevance, 0.0)
            self.assertLessEqual(relevance, 100.0)

    def test_04_gemini_helper_initialization(self) -> None:
        """Verify GeminiHelper initialization and graceful offline/dummy key handling."""
        gemini = GeminiHelper(api_key="test_dummy_key")
        self.assertEqual(gemini.api_key, "test_dummy_key")
        
        # Test prompt structure building without crashing when using dummy key
        response = gemini.generate_grounded_answer(
            user_query="Where are the gates?",
            retrieved_chunks=[{"text": "Gates open 3 hours before kickoff.", "metadata": {"stadium": "Test", "category": "gates"}}]
        )
        self.assertIn("detected_language", response)
        self.assertIn("answer", response)
        self.assertIn("is_grounded", response)


if __name__ == "__main__":
    unittest.main()
