#app.py

import streamlit as st
import os
import hashlib
import pickle
import tempfile
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our custom modules
from pdf_processor import PDFProcessor
from rag_pipeline import RAGPipeline
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="AI Textbook Tutor",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    if "textbook_processed" not in st.session_state:
        st.session_state.textbook_processed = False
    if "textbook_title" not in st.session_state:
        st.session_state.textbook_title = ""
    if "processing_progress" not in st.session_state:
        st.session_state.processing_progress = {}

def get_file_hash(file_content: bytes) -> str:
    """Generate hash for uploaded file to enable caching"""
    return hashlib.md5(file_content).hexdigest()

def save_cache(file_hash: str, rag_pipeline: RAGPipeline, title: str):
    """Save processed data to cache"""
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"{file_hash}.pkl")
    cache_data = {
        'rag_pipeline': rag_pipeline,
        'title': title
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

def load_cache(file_hash: str) -> Dict[str, Any]:
    """Load processed data from cache"""
    cache_file = os.path.join(Config.CACHE_DIR, f"{file_hash}.pkl")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def process_pdf(uploaded_file) -> bool:
    """Process uploaded PDF file with enhanced visual content support"""
    try:
        # Get file content and hash
        file_content = uploaded_file.read()
        file_hash = get_file_hash(file_content)
        
        # Check cache first
        cached_data = load_cache(file_hash)
        if cached_data:
            st.success("📦 Found cached version of this textbook!")
            st.session_state.rag_pipeline = cached_data['rag_pipeline']
            st.session_state.textbook_title = cached_data['title']
            st.session_state.textbook_processed = True
            
            # Show visual content summary if available
            if hasattr(st.session_state.rag_pipeline, 'get_visual_content_summary'):
                visual_summary = st.session_state.rag_pipeline.get_visual_content_summary()
                if visual_summary['total'] > 0:
                    st.info(f"📊 This textbook contains {visual_summary['total']} visual elements (images, diagrams, equations)")
            
            return True
        
        # Process new file
        with st.spinner("📖 Processing your textbook with visual content analysis..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            try:
                # Initialize progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Extract text and images from PDF
                status_text.text("🔍 Extracting text and analyzing visual content...")
                progress_bar.progress(15)
                
                pdf_processor = PDFProcessor()
                text_content, title, visual_content = pdf_processor.extract_text_and_images(tmp_path)
                
                if not text_content.strip():
                    st.error("❌ Could not extract text from the PDF. Please ensure it's a text-based PDF.")
                    return False
                
                # Step 2: Process and chunk text
                status_text.text("✂️ Processing and chunking text content...")
                progress_bar.progress(30)
                
                chunks = pdf_processor.chunk_text(text_content)
                
                if not chunks:
                    st.error("❌ Could not create text chunks. The PDF might be empty or corrupted.")
                    return False
                
                # Step 3: Create RAG pipeline and generate embeddings
                status_text.text("🧠 Creating embeddings for text and visual content...")
                progress_bar.progress(50)
                
                rag_pipeline = RAGPipeline()
                rag_pipeline.create_index(chunks, visual_content)
                
                # Step 4: Save to session state and cache
                status_text.text("💾 Saving processed data...")
                progress_bar.progress(80)
                
                st.session_state.rag_pipeline = rag_pipeline
                st.session_state.textbook_title = title or uploaded_file.name
                st.session_state.textbook_processed = True
                
                # Save to cache for future use
                save_cache(file_hash, rag_pipeline, st.session_state.textbook_title)
                
                # Complete
                progress_bar.progress(100)
                status_text.text("✅ Textbook processed successfully!")
                
                # Show comprehensive stats
                visual_summary = rag_pipeline.get_visual_content_summary()
                
                success_msg = f"""
                🎉 **Enhanced Textbook Processing Complete!**
                - **Title:** {st.session_state.textbook_title}
                - **Text Chunks:** {len(chunks)}
                - **Visual Elements:** {visual_summary['total']} (images, diagrams, equations)
                - **Estimated Pages:** {len(text_content) // 2000}
                """
                
                if visual_summary['total'] > 0:
                    visual_types = ", ".join([f"{count} {type}" for type, count in visual_summary['by_type'].items()])
                    success_msg += f"\n- **Visual Content Types:** {visual_types}"
                
                st.success(success_msg)
                
                # Show visual content breakdown if present
                if visual_summary['total'] > 0:
                    with st.expander("📊 Visual Content Analysis"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**By Content Type:**")
                            for content_type, count in visual_summary['by_type'].items():
                                st.write(f"- {content_type.title()}: {count}")
                        
                        with col2:
                            st.write("**Pages with Visual Content:**")
                            pages_with_visuals = len(visual_summary['by_page'])
                            st.write(f"- {pages_with_visuals} pages contain visual elements")
                            if pages_with_visuals <= 10:
                                page_list = ", ".join(map(str, sorted(visual_summary['by_page'].keys())))
                                st.write(f"- Pages: {page_list}")
                
                return True
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        st.error(f"❌ Error processing PDF: {str(e)}")
        return False

def display_chat_interface():
    """Display the chat interface"""
    st.subheader(f"💬 Chat with: {st.session_state.textbook_title}")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📄 Source References"):
                    for i, source in enumerate(message["sources"], 1):
                        st.text(f"Source {i}: {source[:200]}...")

def handle_user_input():
    """Handle user question input with visual content support"""
    if prompt := st.chat_input("Ask a question about your textbook..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response with visual content
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing text and visual content..."):
                try:
                    response, sources, visual_info = st.session_state.rag_pipeline.answer_question(prompt)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Show enhanced sources with visual indicators
                    if sources:
                        with st.expander("📄 Source References"):
                            text_sources = []
                            visual_sources = []
                            
                            for i, source in enumerate(sources, 1):
                                if source.startswith("[Visual Content"):
                                    visual_sources.append((i, source))
                                else:
                                    text_sources.append((i, source))
                            
                            # Display text sources
                            if text_sources:
                                st.write("**📝 Text Sources:**")
                                for i, source in text_sources:
                                    st.text(f"{i}. {source[:200]}...")
                            
                            # Display visual sources
                            if visual_sources:
                                st.write("**🖼️ Visual Content Sources:**")
                                for i, source in visual_sources:
                                    st.text(f"{i}. {source[:200]}...")
                    
                    # Show visual content summary if present
                    if visual_info.get("has_visual", False):
                        with st.expander("📊 Visual Content Used"):
                            st.write(f"This answer incorporates {visual_info['visual_count']} visual elements:")
                            for visual_type in set(visual_info['visual_types']):
                                count = visual_info['visual_types'].count(visual_type)
                                st.write(f"- {visual_type}: {count}")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources,
                        "visual_info": visual_info
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

def display_sidebar():
    """Display sidebar with controls and information"""
    with st.sidebar:
        st.title("📚 AI Textbook Tutor")
        st.markdown("Upload a textbook PDF and start learning!")
        
        st.markdown("---")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF textbook",
            type=['pdf'],
            help="Upload a textbook in PDF format"
        )
        
        if uploaded_file is not None:
            if st.button("📖 Process Textbook", type="primary"):
                if process_pdf(uploaded_file):
                    st.rerun()
        
        st.markdown("---")
        
        # Show textbook info if processed
        if st.session_state.textbook_processed:
            st.success("✅ Textbook Ready")
            st.info(f"**Current Book:** {st.session_state.textbook_title}")
            
            # Show visual content statistics
            if hasattr(st.session_state.rag_pipeline, 'get_visual_content_summary'):
                visual_summary = st.session_state.rag_pipeline.get_visual_content_summary()
                if visual_summary['total'] > 0:
                    with st.expander("📊 Visual Content Summary"):
                        st.write(f"**Total Elements:** {visual_summary['total']}")
                        
                        if visual_summary['by_type']:
                            st.write("**Content Types:**")
                            for content_type, count in visual_summary['by_type'].items():
                                icon = "📐" if content_type == "mathematical" else "📊" if content_type == "chart" else "🖼️"
                                st.write(f"{icon} {content_type.title()}: {count}")
            
            # Clear chat history button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
            
            # Reset textbook button
            if st.button("🔄 Upload New Textbook"):
                st.session_state.textbook_processed = False
                st.session_state.rag_pipeline = None
                st.session_state.messages = []
                st.session_state.textbook_title = ""
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Enhanced Features:
        - **Visual Content Analysis**: Processes images, diagrams, and equations
        - **Mathematical Equations**: Describes and explains formulas
        - **Chart & Graph Analysis**: Interprets data visualizations
        - **Comprehensive Search**: Finds both text and visual information
        
        ### 🤔 Example Questions:
        - "Explain the diagram in Chapter 3"
        - "What does the equation on page 45 mean?"
        - "Describe the chart showing sales data"
        - "What are the key concepts illustrated?"
        """)
        
        st.markdown("---")
        st.markdown("*Enhanced with Visual AI & Gemini 1.5*")

def main():
    """Main application function"""
    setup_page_config()
    initialize_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    if not st.session_state.textbook_processed:
        # Welcome screen
        st.title("📚 AI Textbook Tutor")
        st.markdown("""
        Welcome to your personal AI tutor! This application helps you learn from any textbook by:
        
        🔍 **Smart Content Search** - Finds relevant sections instantly  
        🤖 **AI-Powered Explanations** - Provides clear, textbook-based answers  
        💬 **Interactive Chat** - Ask questions in natural language  
        📄 **Source References** - Shows exactly where information comes from  
        
        ### How to get started:
        1. **Upload** your textbook PDF using the sidebar
        2. **Process** the textbook (this may take a few minutes)
        3. **Start asking** questions about the content!
        
        ### Example questions you can ask:
        - "Explain the concept of photosynthesis"
        - "What are the key points in Chapter 5?"
        - "Give me an example of supply and demand"
        - "How do I solve quadratic equations?"
        """)
        
        # Show example interface
        st.markdown("---")
        st.subheader("📱 Preview of Chat Interface")
        
        with st.chat_message("user"):
            st.write("What is the main theme of Chapter 3?")
        
        with st.chat_message("assistant"):
            st.write("""
            Based on Chapter 3 of your textbook, the main theme is **Economic Growth and Development**. 
            The chapter covers three key areas:
            
            1. **Factors of Production** - How land, labor, and capital contribute to growth
            2. **Technological Innovation** - The role of technology in economic advancement  
            3. **Government Policy** - How fiscal and monetary policies affect development
            
            The chapter emphasizes that sustainable growth requires balancing these elements...
            """)
            
            with st.expander("📄 Source References"):
                st.text("Source 1: Chapter 3, Section 3.1 - Introduction to Economic Growth...")
                st.text("Source 2: Chapter 3, Section 3.2 - Factors Affecting Development...")
    
    else:
        # Chat interface
        display_chat_interface()
        handle_user_input()

if __name__ == "__main__":
    main()