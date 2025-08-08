import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Improved configuration settings for much better retrieval performance"""
    
    # API Configuration - support both variable names
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY', '')
    
    # File paths
    BASE_DIR = Path(__file__).parent
    CACHE_DIR = BASE_DIR / 'cache'
    TEMP_DIR = BASE_DIR / 'temp'
    
    # Improved embedding settings
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    EMBEDDING_DIM = 384
    
    # Optimized text processing settings for better content preservation
    CHUNK_SIZE = 1200  # Larger chunks for better context
    CHUNK_OVERLAP = 200  # More overlap for better continuity
    MAX_CHUNKS_FOR_CONTEXT = 12  # More chunks for comprehensive answers
    
    # Much more permissive similarity thresholds for better retrieval
    SIMILARITY_THRESHOLD = -2.0  # Very permissive threshold
    VISUAL_SIMILARITY_THRESHOLD = -2.0  # Very permissive for visual content
    
    # Enhanced retrieval settings
    INITIAL_RETRIEVAL_K = 20  # Get more candidates initially
    FINAL_RETRIEVAL_K = 12  # Return more results after filtering
    
    # Gemini settings
    GEMINI_MODEL = 'gemini-1.5-flash'
    MAX_TOKENS = 4096  # Increased for longer responses
    TEMPERATURE = 0.3
    
    # UI settings
    PAGE_TITLE = "AI Textbook Tutor"
    PAGE_ICON = "📚"
    
    # Processing settings for better content preservation
    MIN_CHUNK_LENGTH = 30  # Lower minimum for better coverage
    MIN_CHUNK_WORDS = 3    # Very low minimum
    PRESERVE_HEADERS = True
    AGGRESSIVE_CLEANING = False  # Disable aggressive cleaning
    
    # Enhanced keyword and concept extraction settings
    ENABLE_KEYWORD_INDEXING = True
    ENABLE_CHAPTER_INDEXING = True
    ENABLE_CONCEPT_INDEXING = True
    KEYWORD_BOOST_SCORE = 0.9  # High boost for keyword matches
    
    # Visual content settings
    PROCESS_VISUAL_CONTENT = True
    OCR_ENABLED = True
    VISUAL_DESCRIPTION_LENGTH = 300  # Longer descriptions
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.CACHE_DIR.mkdir(exist_ok=True)
        cls.TEMP_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def is_gemini_configured(cls) -> bool:
        """Check if Gemini API key is configured"""
        return bool(cls.GEMINI_API_KEY and cls.GEMINI_API_KEY.strip())
    
    @classmethod
    def get_env_template(cls) -> str:
        """Get template for .env file"""
        return """# AI Tutor Environment Configuration
# Google Gemini API Key (required)
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Adjust model settings
GEMINI_MODEL=gemini-1.5-flash
MAX_TOKENS=4096
TEMPERATURE=0.3

# Optional: Adjust retrieval settings (improved defaults)
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
MAX_CHUNKS_FOR_CONTEXT=12
SIMILARITY_THRESHOLD=-2.0
"""
    
    @classmethod
    def get_debug_info(cls) -> dict:
        """Get configuration debug information"""
        return {
            "api_configured": cls.is_gemini_configured(),
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "max_chunks": cls.MAX_CHUNKS_FOR_CONTEXT,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "visual_threshold": cls.VISUAL_SIMILARITY_THRESHOLD,
            "initial_retrieval_k": cls.INITIAL_RETRIEVAL_K,
            "final_retrieval_k": cls.FINAL_RETRIEVAL_K,
            "embedding_model": cls.EMBEDDING_MODEL,
            "keyword_indexing": cls.ENABLE_KEYWORD_INDEXING,
            "chapter_indexing": cls.ENABLE_CHAPTER_INDEXING,
            "concept_indexing": cls.ENABLE_CONCEPT_INDEXING,
            "keyword_boost": cls.KEYWORD_BOOST_SCORE
        }
    
    @classmethod
    def get_performance_settings(cls) -> dict:
        """Get performance-related settings"""
        return {
            "chunk_settings": {
                "size": cls.CHUNK_SIZE,
                "overlap": cls.CHUNK_OVERLAP,
                "min_length": cls.MIN_CHUNK_LENGTH,
                "min_words": cls.MIN_CHUNK_WORDS
            },
            "retrieval_settings": {
                "similarity_threshold": cls.SIMILARITY_THRESHOLD,
                "initial_k": cls.INITIAL_RETRIEVAL_K,
                "final_k": cls.FINAL_RETRIEVAL_K,
                "max_context_chunks": cls.MAX_CHUNKS_FOR_CONTEXT
            },
            "indexing_features": {
                "keyword_indexing": cls.ENABLE_KEYWORD_INDEXING,
                "chapter_indexing": cls.ENABLE_CHAPTER_INDEXING,
                "concept_indexing": cls.ENABLE_CONCEPT_INDEXING,
                "keyword_boost": cls.KEYWORD_BOOST_SCORE
            },
            "generation_settings": {
                "model": cls.GEMINI_MODEL,
                "max_tokens": cls.MAX_TOKENS,
                "temperature": cls.TEMPERATURE
            }
        }