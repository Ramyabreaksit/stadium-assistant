# ⚽ Stadium Assistant — Multilingual FIFA World Cup 2026 RAG Chatbot

**Stadium Assistant** is an intelligent, multilingual web application built with **Streamlit**, **Google Gemini API**, and **ChromaDB** (Retrieval-Augmented Generation) to help fans navigate the **FIFA World Cup 2026**.

Fans can ask questions in **any language** (English, Spanish, Hindi, French, Arabic, Japanese, etc.) about:
- 🗺️ **Stadium Navigation & Gates**
- 💺 **Seating Tiers & Sections**
- 🍔 **Food & Concessions** (Halal, Vegan, Dietary requirements)
- 🚻 **Restrooms & Family Care**
- 🚆 **Public Transport, Shuttles & Parking**
- 🎒 **Bag Policies & Prohibited Items**
- 🚪 **Emergency Exits & First Aid**

The app automatically detects the user's language, searches the local ChromaDB vector database (`stadium_faq.json` / markdown docs) using multilingual semantic embeddings, and generates accurate, friendly answers grounded strictly in factual stadium information using Google's Gemini AI models (`gemini-2.5-flash` / `gemini-2.5-pro`).

---

## 🚀 Features

- **Multilingual Support & Automatic Language Detection**: Powered by the Gemini API (`google-genai` SDK). Ask in Hindi, receive an accurate answer in Hindi grounded in English/multilingual stadium data!
- **Local RAG with ChromaDB**: Vector retrieval ensures no hallucinations—answers are strictly cited from official stadium FAQ documents.
- **Sleek UI & Glassmorphism Design**: FIFA World Cup 2026 branding, stadium venue selector filters, quick question chips, language indicator badges, and expandable grounding sources.
- **Easy Customization**: Drop your own FAQ `.json` or `.md` files directly into `knowledge_base/data/` or `knowledge_base/docs/` and re-index from the sidebar.
- **Cloud Run Ready**: Includes optimized `Dockerfile` and `requirements.txt` pre-loaded with embedding weights for instant cold start times on Google Cloud Run.

---

## 🛠️ Local Setup & Running

### 1. Prerequisites
- Python 3.11+
- A Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/)

### 2. Installation

Clone or open the repository directory:
```bash
cd stadium_assistant
```

Create a virtual environment and install dependencies:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Set Environment Variable
Set your Gemini API key in your terminal or create a `.env` file:
```bash
# Windows PowerShell:
$env:GEMINI_API_KEY="your-api-key-here"

# Linux / macOS:
export GEMINI_API_KEY="your-api-key-here"
```
*(Alternatively, you can input your API key directly in the Streamlit sidebar inside the app).*

### 4. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deploying to Google Cloud Run

### Using Google Cloud SDK (`gcloud`)

1. **Authenticate and set project**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_GCP_PROJECT_ID
   ```

2. **Deploy directly from source**:
   ```bash
   gcloud run deploy stadium-assistant \
     --source . \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars="GEMINI_API_KEY=your-api-key-here" \
     --memory 2Gi \
     --cpu 2
   ```

### Using Docker locally or for Cloud Build

1. **Build Docker image**:
   ```bash
   docker build -t stadium-assistant .
   ```

2. **Run Docker container locally**:
   ```bash
   docker run -p 8080:8080 -e GEMINI_API_KEY="your-api-key-here" stadium-assistant
   ```
   Access at `http://localhost:8080`.

---

## 📁 Project Structure

```
stadium_assistant/
├── app.py                     # Main Streamlit UI interface
├── style.css                  # Custom FIFA World Cup 2026 glassmorphism styling
├── gemini_helper.py           # Gemini API client, language detection & RAG generation
├── rag_engine.py              # ChromaDB client, embedding model, & semantic search
├── requirements.txt           # Python package dependencies
├── Dockerfile                 # Optimized Cloud Run Docker configuration
├── .dockerignore              # Files excluded from Docker build
├── knowledge_base/
│   ├── ingest.py              # Ingestion engine parsing JSON/Markdown into ChromaDB
│   └── data/
│       └── stadium_faq.json   # Comprehensive FAQ data for MetLife, SoFi, Estadio Azteca & General Rules
└── README.md                  # Project documentation
```

---

## 📚 Adding Custom FAQ Data

To add new stadium details or custom rules:
1. Open `knowledge_base/data/stadium_faq.json` and add entries with fields: `stadium`, `category`, `question`, and `answer`.
2. Or drop `.md` or `.txt` files into `knowledge_base/docs/`.
3. In the app sidebar, click **"🔄 Re-Index Knowledge Base"** to instantly update ChromaDB.
