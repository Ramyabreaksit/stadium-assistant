import os
import json
import glob
from typing import List, Dict, Any

def parse_json_faq(file_path: str) -> List[Dict[str, Any]]:
    """Parse JSON FAQ file into standard document structure for ChromaDB."""
    documents = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    doc_id = item.get("id", f"{os.path.basename(file_path)}_{idx}")
                    stadium = item.get("stadium", "General")
                    category = item.get("category", "general")
                    question = item.get("question", "")
                    answer = item.get("answer", "")
                    
                    # Combine question and answer for rich semantic embedding search
                    full_text = f"Stadium: {stadium}\nCategory: {category.upper()}\nQuestion: {question}\nAnswer: {answer}"
                    
                    documents.append({
                        "id": doc_id,
                        "text": full_text,
                        "metadata": {
                            "stadium": stadium,
                            "category": category,
                            "question": question,
                            "answer": answer,
                            "source_file": os.path.basename(file_path)
                        }
                    })
    except Exception as e:
        print(f"Error parsing JSON file {file_path}: {e}")
    return documents

def parse_markdown_faq(file_path: str) -> List[Dict[str, Any]]:
    """Parse Markdown or Text FAQ files into chunks."""
    documents = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Split by headings (`## ` or `### `) if structured, otherwise chunk by paragraphs
            sections = content.split("\n## ")
            stadium_name = os.path.splitext(os.path.basename(file_path))[0].replace("_", " ").title()
            
            for idx, section in enumerate(sections):
                if not section.strip():
                    continue
                lines = section.strip().split("\n")
                heading = lines[0].replace("#", "").strip() if idx > 0 else "Overview"
                body = "\n".join(lines[1:] if idx > 0 else lines).strip()
                if not body:
                    body = heading
                    
                doc_id = f"{os.path.splitext(os.path.basename(file_path))[0]}_sec_{idx}"
                full_text = f"Stadium/Doc: {stadium_name}\nSection: {heading}\nContent: {body}"
                
                documents.append({
                    "id": doc_id,
                    "text": full_text,
                    "metadata": {
                        "stadium": stadium_name,
                        "category": "general_doc",
                        "question": heading,
                        "answer": body,
                        "source_file": os.path.basename(file_path)
                    }
                })
    except Exception as e:
        print(f"Error parsing Markdown file {file_path}: {e}")
    return documents

def load_all_faq_documents(data_dir: str = None) -> List[Dict[str, Any]]:
    """Scan the data directory and load all JSON, Markdown, and TXT files."""
    if data_dir is None:
        # Default to directory containing this script -> data/
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        
    all_documents = []
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        return all_documents
        
    # Find all supported files
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    md_files = glob.glob(os.path.join(data_dir, "*.md")) + glob.glob(os.path.join(data_dir, "*.txt"))
    
    for jf in json_files:
        all_documents.extend(parse_json_faq(jf))
        
    for mf in md_files:
        all_documents.extend(parse_markdown_faq(mf))
        
    print(f"Successfully loaded {len(all_documents)} FAQ chunks from {data_dir}.")
    return all_documents

if __name__ == "__main__":
    docs = load_all_faq_documents()
    print(f"Sample loaded document: {docs[0] if docs else 'None'}")
