# 📚 AI Textbook Tutor

An interactive **PDF-based learning assistant** that lets you upload a textbook and chat with it like an instructor.  
The system uses **Retrieval-Augmented Generation (RAG)** to search the most relevant parts of your uploaded book and generate answers — including explanations of **text, diagrams, and mathematical content** — using **Google Gemini 1.5**.

---

## 🔍 Features

* **PDF Processing:** Extracts and cleans text, pulls diagrams, charts, and equations from pages, and uses OCR (**Tesseract**) for text inside images.
* **Semantic Search:** Uses FAISS vector search over both text and AI-generated image descriptions, with context-aware chunking for better retrieval.
* **AI Answering:** Combines relevant passages and visual context to provide tutor-style explanations grounded in your book, including page and source references.
* **UI:** A **Streamlit** interface for uploading, processing, and chatting, with caching for previously processed PDFs.

---

## 🛠️ Tech Stack

| Component | Tool |
| :--- | :--- |
| **Frontend** | Streamlit |
| **PDF & Image Processing** | PyMuPDF, Pillow, OpenCV |
| **OCR** | Tesseract |
| **Embeddings & Search** | `sentence-transformers`, FAISS |
| **LLM** | Google Gemini 1.5 (Vision + Text) |
| **Environment** | Python 3.8+, `dotenv` |

---

## 📦 Installation

### **Prerequisites**

* Python **3.8+**
* A [Google Gemini API key](https://makersuite.google.com/app/apikey)
* **Tesseract OCR** installed and in your system's PATH.

### **Install Tesseract**

**Windows:** [Download here](https://github.com/UB-Mannheim/tesseract/wiki) and add it to your system PATH.

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr
```

### **🚀 Steps to Run**
1. Clone the repository:
```bash
git clone <repository-url>
cd ai-textbook-tutor
```
2. Install the dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file in the project's root directory and add your API key:
```env
GEMINI_API_KEY=your_api_key_here
```
4. Run the app:
```bash
streamlit run app.py
```

---

## 📖 How to Use

1.  **Upload a PDF textbook:** Drag and drop your PDF into the upload section of the Streamlit app. This can be a text-based or scanned document.
2.  **Processing:** The app will process the PDF by splitting text into searchable chunks, classifying and describing images, and running OCR on scanned pages.
3.  **Ask Questions:** Type your question in the chat input field. The system will retrieve relevant content from your book and use Google Gemini to generate a grounded answer, including source page numbers.

**Example Usage**
```
Q: Explain the diagram on page 15 showing cell division
A: The diagram shows four stages of mitosis: prophase, metaphase, anaphase, and telophase...
```

**Tips for Best Results**
* Use precise page references if you want to discuss a specific diagram or figure.
* Break down complex questions into smaller parts for better retrieval accuracy.
* Re-upload the PDF if you update or replace the textbook.

---

## ⚠️ Current Limitations

* Retrieval can be less accurate for very large books or long, complex questions.
* Free-tier users may be limited by token limits (250k input tokens/minute) and rate limits.
* Initial processing of large books can take several minutes.

---

## 🗺️ Planned Improvements

* Smarter retrieval and chunking strategies.
* Enhanced output formatting with clearer structure and highlights.
* UI enhancements, such as chat history and page previews.
* Better support for books with over 500 pages.

### **🤖 Future AI Agent Features**

We plan to extend this into a multi-agent academic assistant with the following agents:
* **Teaching Assistant (TA) Agent:** Generates practice assignments, evaluates answers, and gives feedback.
* **Progress Report Agent:** Tracks sessions, sends email summaries, and highlights strengths/weaknesses.
* **Learning Plan Agent:** Creates structured learning plans that adapt as you progress.
* **Quiz & Test Agent:** Generates quizzes and provides instant scoring with explanations.

---

## 🤝 Contributing

We welcome improvements, especially for:
* Retrieval quality
* Token-efficient summarization for large PDFs
* UI/UX for studying
* Multi-agent orchestration
