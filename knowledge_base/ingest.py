import os
import json
import glob
from typing import List, Dict, Any, Optional


def parse_json_faq(file_path: str) -> List[Dict[str, Any]]:
    """Parse a structured JSON FAQ file into standardized dictionary chunks for ChromaDB indexing.

    Args:
        file_path (str): Absolute or relative path to the `.json` file containing FAQ items.

    Returns:
        List[Dict[str, Any]]: A list of document dictionaries formatted with `'id'`, `'text'`,
        and `'metadata'` (`stadium`, `category`, `question`, `answer`, `source_file`).
    """
    documents: List[Dict[str, Any]] = []
    if not os.path.exists(file_path):
        print(f"[Error] FAQ JSON file not found: {file_path}")
        return documents

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    if not isinstance(item, dict):
                        continue
                    doc_id = str(item.get("id", f"{os.path.basename(file_path)}_{idx}"))
                    stadium = str(item.get("stadium", "General"))
                    category = str(item.get("category", "general"))
                    question = str(item.get("question", ""))
                    answer = str(item.get("answer", ""))
                    
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
            else:
                print(f"[Warning] Expected a list of JSON items in {file_path}, but received {type(data)}.")
    except json.JSONDecodeError as jde:
        print(f"[Error] Failed to parse JSON content from {file_path}: {jde}")
    except OSError as ose:
        print(f"[Error] I/O error while reading JSON file {file_path}: {ose}")
    except Exception as e:
        print(f"[Error] Unexpected error processing JSON file {file_path}: {e}")
    return documents


def parse_markdown_faq(file_path: str) -> List[Dict[str, Any]]:
    """Parse structured Markdown (`.md`) or text (`.txt`) FAQ files into section chunks.

    Args:
        file_path (str): Absolute or relative path to the Markdown or Text document.

    Returns:
        List[Dict[str, Any]]: A list of document dictionaries containing text chunks and metadata.
    """
    documents: List[Dict[str, Any]] = []
    if not os.path.exists(file_path):
        print(f"[Error] FAQ Markdown file not found: {file_path}")
        return documents

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Split by headings (`## `) if structured, otherwise chunk by sections
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
    except OSError as ose:
        print(f"[Error] I/O error while reading Markdown file {file_path}: {ose}")
    except Exception as e:
        print(f"[Error] Unexpected error processing Markdown file {file_path}: {e}")
    return documents


def load_all_faq_documents(data_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Scan the target knowledge base directory and load all supported `.json`, `.md`, and `.txt` FAQ files.

    Args:
        data_dir (Optional[str]): Absolute or relative path to the directory containing FAQ documents.
            If `None`, defaults to the `data/` subdirectory next to this script.

    Returns:
        List[Dict[str, Any]]: A combined list of all parsed document chunks across files.
    """
    if data_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        
    all_documents: List[Dict[str, Any]] = []
    
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir, exist_ok=True)
        except OSError as ose:
            print(f"[Error] Could not access or create data directory {data_dir}: {ose}")
        return all_documents
        
    try:
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        md_files = glob.glob(os.path.join(data_dir, "*.md")) + glob.glob(os.path.join(data_dir, "*.txt"))
        
        for jf in json_files:
            all_documents.extend(parse_json_faq(jf))
            
        for mf in md_files:
            all_documents.extend(parse_markdown_faq(mf))
            
        print(f"Successfully loaded {len(all_documents)} FAQ chunks from {data_dir}.")
    except Exception as e:
        print(f"[Error] Exception occurred while loading FAQ documents from {data_dir}: {e}")
        
    return all_documents


if __name__ == "__main__":
    loaded_docs = load_all_faq_documents()
    print(f"Sample loaded document: {loaded_docs[0] if loaded_docs else 'None'}")
