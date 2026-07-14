import unittest
import os
import shutil
from knowledge_base.ingest import load_all_faq_documents
from rag_engine import StadiumRAG
from gemini_helper import GeminiHelper

class TestStadiumAssistant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_path = "test_chroma_db"
        if os.path.exists(cls.test_db_path):
            shutil.rmtree(cls.test_db_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db_path):
            try:
                shutil.rmtree(cls.test_db_path)
            except Exception:
                pass

    def test_01_load_faq_documents(self):
        docs = load_all_faq_documents()
        self.assertGreater(len(docs), 0, "Should load at least some FAQ chunks from stadium_faq.json")
        sample = docs[0]
        self.assertIn("id", sample)
        self.assertIn("text", sample)
        self.assertIn("metadata", sample)
        self.assertIn("stadium", sample["metadata"])

    def test_02_rag_indexing_and_retrieval(self):
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

    def test_03_gemini_helper_initialization(self):
        gemini = GeminiHelper(api_key="test_dummy_key")
        self.assertEqual(gemini.api_key, "test_dummy_key")
        
        # Test prompt structure building without failing when offline / dummy key
        response = gemini.generate_grounded_answer(
            user_query="Where are the gates?",
            retrieved_chunks=[{"text": "Gates open 3 hours before kickoff.", "metadata": {"stadium": "Test", "category": "gates"}}]
        )
        self.assertIn("detected_language", response)
        self.assertIn("answer", response)

if __name__ == "__main__":
    unittest.main()
