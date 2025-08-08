# 📚 AI Textbook Tutor

An interactive **PDF-based learning assistant** that lets you upload a textbook and chat with it like an instructor.  
The system uses **Retrieval-Augmented Generation (RAG)** to search the most relevant parts of your uploaded book and generate answers — including explanations of **text, diagrams, and mathematical content** — using **Google Gemini 1.5**.

---

## 🔍 Features

### **PDF Processing**
- Extracts and cleans text
- Pulls diagrams, charts, and equations from pages
- Uses OCR (**Tesseract**) for text inside images

### **Semantic Search**
- FAISS vector search over both text and AI-generated image descriptions
- Context-aware chunking for better retrieval

### **AI Answering**
- Combines relevant passages and visual context
- Tutor-style explanations grounded in your book
- Includes page/source references where possible

### **UI**
- **Streamlit** interface for uploading, processing, and chatting
- Caching for previously processed PDFs

---

## 🛠️ Tech Stack

| Component                | Tool |
|--------------------------|------|
| **Frontend**             | Streamlit |
| **PDF & Image Processing** | PyMuPDF, Pillow, OpenCV |
| **OCR**                  | Tesseract |
| **Embeddings & Search**  | sentence-transformers, FAISS |
| **LLM**                  | Google Gemini 1.5 (Vision + Text) |
| **Environment**          | Python 3.8+, dotenv for config |

---

## 📦 Installation

### **Prerequisites**
- Python **3.8+**
- [Google Gemini API key](https://makersuite.google.com/app/apikey)
- **Tesseract OCR** installed & in system PATH

**Install Tesseract:**

**Windows:**  
[Download here](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH

**macOS:**
```bash
brew install tesseract
Ubuntu/Debian:

bash
Copy
Edit
sudo apt install tesseract-ocr
🚀 Steps to Run
bash
Copy
Edit
git clone <repository-url>
cd ai-textbook-tutor
pip install -r requirements.txt
Create a .env file in the project root:

env
Copy
Edit
GEMINI_API_KEY=your_api_key_here
Run the app:

bash
Copy
Edit
streamlit run app.py
📖 How to Use
Upload a PDF textbook
Can be text-based or scanned (OCR will run for scanned content)

Processing

Text is split into searchable chunks

Images are classified, described, and added to the search index

Ask Questions

The system retrieves top matches and sends them to Gemini

Answer is generated using both text and visual context

Example:

vbnet
Copy
Edit
Q: Explain the diagram on page 15 showing cell division
A: The diagram shows four stages...
📂 Project Structure
bash
Copy
Edit
ai-textbook-tutor/
├── app.py             # Streamlit interface
├── pdf_processor.py   # PDF + image processing
├── rag_pipeline.py    # RAG search + answer generation
├── config.py          # Settings
├── requirements.txt   # Dependencies
├── cache/             # Cached PDFs
└── temp/              # Temp files
⚠️ Current Limitations
Retrieval can be less accurate for:

Very large books on free-tier Gemini (token limits)

Long, complex, or multi-part questions

Free tier rate limits (250k input tokens/minute)

Initial processing of large books can take several minutes

🗺️ Planned Improvements
Smarter retrieval & chunking

Output formatting with clearer structure & highlights

UI enhancements (answer history, page previews)

Option to choose different LLM backends

Better support for 500+ page books without hitting token limits

🤖 Future AI Agent Features
We plan to extend this into a multi-agent academic assistant.

Upcoming agents:

Teaching Assistant (TA) Agent – Generates practice assignments, evaluates answers, and gives feedback

Progress Report Agent – Tracks sessions, sends email summaries, highlights strengths/weaknesses

Learning Plan Agent – Creates structured learning plans, adapts as you progress

Plan Execution Agent – Guides daily study, checks progress

Quiz & Test Agent – Generates quizzes, provides instant scoring with explanations

🤝 Contributing
We welcome improvements, especially for:

Retrieval quality

Token-efficient summarisation for large PDFs

UI/UX for studying

Multi-agent orchestration