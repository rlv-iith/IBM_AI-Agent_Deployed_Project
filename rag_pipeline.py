import numpy as np
import faiss
import logging
from typing import List, Tuple, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from config import Config
import re
from collections import Counter

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Significantly improved RAG Pipeline with better retrieval and context understanding"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_dim = 384
        self.index = None
        self.text_chunks = []
        self.visual_content = []
        self.chunk_metadata = []
        self.processed_chunks = []
        
        # Create keyword index for better retrieval
        self.keyword_index = {}
        self.chapter_index = {}
        self.concept_index = {}
        
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Enhanced RAG Pipeline initialized with improved retrieval")

    def _extract_keywords_and_concepts(self, text: str) -> Dict[str, List[str]]:
        """Extract keywords, chapters, sections, and mathematical concepts"""
        text_lower = text.lower()
        
        # Extract chapter and section references
        chapters = re.findall(r'chapter\s+(\d+)', text_lower)
        sections = re.findall(r'section\s+(\d+(?:\.\d+)?)', text_lower)
        
        # Extract mathematical concepts
        math_concepts = []
        math_patterns = [
            r'theorem\s+\d+', r'definition\s+\d+', r'lemma\s+\d+', r'proof',
            r'algorithm', r'formula', r'equation', r'function', r'graph',
            r'vertex', r'edge', r'tree', r'matrix', r'vector', r'set',
            r'probability', r'combinatorial', r'discrete', r'logic',
            r'induction', r'recursion', r'complexity', r'optimization'
        ]
        
        for pattern in math_patterns:
            matches = re.findall(pattern, text_lower)
            math_concepts.extend(matches)
        
        # Extract important terms (capitalized words, definitions)
        important_terms = []
        # Look for definition patterns
        def_patterns = [
            r'(\w+)\s+is\s+defined\s+as',
            r'(\w+)\s+are\s+defined\s+as',
            r'a\s+(\w+)\s+is\s+a',
            r'an\s+(\w+)\s+is\s+a',
            r'the\s+(\w+)\s+of\s+a'
        ]
        
        for pattern in def_patterns:
            matches = re.findall(pattern, text_lower)
            important_terms.extend(matches)
        
        # Extract capitalized terms (likely important concepts)
        cap_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        important_terms.extend([term.lower() for term in cap_terms if len(term) > 3])
        
        return {
            'chapters': chapters,
            'sections': sections,
            'math_concepts': math_concepts,
            'important_terms': list(set(important_terms))
        }

    def _build_keyword_indices(self):
        """Build keyword indices for faster retrieval"""
        self.keyword_index = {}
        self.chapter_index = {}
        self.concept_index = {}
        
        for i, chunk_meta in enumerate(self.chunk_metadata):
            content = chunk_meta['original_content'].lower()
            
            # Extract and index keywords
            keywords = self._extract_keywords_and_concepts(chunk_meta['original_content'])
            
            # Index by chapters
            for chapter in keywords['chapters']:
                if chapter not in self.chapter_index:
                    self.chapter_index[chapter] = []
                self.chapter_index[chapter].append(i)
            
            # Index by mathematical concepts
            for concept in keywords['math_concepts']:
                if concept not in self.concept_index:
                    self.concept_index[concept] = []
                self.concept_index[concept].append(i)
            
            # Index by important terms
            for term in keywords['important_terms']:
                if term not in self.keyword_index:
                    self.keyword_index[term] = []
                self.keyword_index[term].append(i)
            
            # Also index by common words for broader matching
            words = re.findall(r'\b\w{4,}\b', content)  # Words with 4+ characters
            word_counts = Counter(words)
            
            for word, count in word_counts.most_common(20):  # Top 20 words per chunk
                if word not in self.keyword_index:
                    self.keyword_index[word] = []
                self.keyword_index[word].append(i)

    def create_index(self, text_chunks: List[str], visual_content: List[Dict[str, Any]] = None):
        """Create index with much better preprocessing and indexing"""
        try:
            self.text_chunks = text_chunks
            self.visual_content = visual_content or []
            
            content_for_embedding = []
            self.chunk_metadata = []
            self.processed_chunks = []
            
            logger.info(f"Processing {len(text_chunks)} text chunks with improved indexing...")
            
            # Process text chunks with minimal modification to preserve content
            for i, chunk in enumerate(text_chunks):
                # Keep original content mostly unchanged for better retrieval
                original_chunk = chunk.strip()
                
                # Only light enhancement - preserve original content structure
                enhanced_chunk = self._lightly_enhance_for_embedding(original_chunk)
                
                content_for_embedding.append(enhanced_chunk)
                self.processed_chunks.append(enhanced_chunk)
                
                self.chunk_metadata.append({
                    'type': 'text',
                    'index': i,
                    'original_content': original_chunk,
                    'enhanced_content': enhanced_chunk,
                    'length': len(original_chunk),
                    'word_count': len(original_chunk.split())
                })
            
            # Process visual content
            for i, visual in enumerate(self.visual_content):
                visual_text = self._create_visual_description(visual)
                content_for_embedding.append(visual_text)
                self.processed_chunks.append(visual_text)
                
                self.chunk_metadata.append({
                    'type': 'visual',
                    'index': i,
                    'original_content': visual_text,
                    'enhanced_content': visual_text,
                    'visual_data': visual,
                    'length': len(visual_text),
                    'word_count': len(visual_text.split())
                })
            
            # Build keyword indices for fast lookup
            self._build_keyword_indices()
            
            logger.info(f"Built keyword indices: {len(self.keyword_index)} terms, {len(self.chapter_index)} chapters, {len(self.concept_index)} concepts")
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(content_for_embedding)} items...")
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(content_for_embedding), batch_size):
                batch = content_for_embedding[i:i+batch_size]
                batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
                all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings).astype('float32')
            faiss.normalize_L2(embeddings)
            
            # Create FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(embeddings)
            
            logger.info(f"FAISS index created with {self.index.ntotal} vectors")
            logger.info(f"Index complete: {len(text_chunks)} text chunks, {len(self.visual_content)} visual items")
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            raise e

    def _lightly_enhance_for_embedding(self, text: str) -> str:
        """Light enhancement that preserves original content"""
        # Just clean up whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        return text

    def _create_visual_description(self, visual: Dict[str, Any]) -> str:
        """Create searchable description for visual content"""
        description = visual.get('description', '')
        page = visual.get('page', 0) + 1
        content_type = visual.get('type', 'visual')
        
        enhanced_desc = f"Visual content from page {page}: {description} (Type: {content_type})"
        return enhanced_desc

    def _find_keyword_matches(self, query: str) -> List[int]:
        """Find chunks that match query keywords"""
        query_lower = query.lower()
        matched_indices = set()
        
        # Extract query terms
        query_words = re.findall(r'\b\w{3,}\b', query_lower)
        
        # Look for exact matches in our indices
        for word in query_words:
            if word in self.keyword_index:
                matched_indices.update(self.keyword_index[word])
            
            # Partial matches for important terms
            for indexed_term in self.keyword_index:
                if word in indexed_term or indexed_term in word:
                    matched_indices.update(self.keyword_index[indexed_term])
        
        # Special handling for chapters and sections
        chapters = re.findall(r'chapter\s+(\d+)', query_lower)
        for chapter in chapters:
            if chapter in self.chapter_index:
                matched_indices.update(self.chapter_index[chapter])
        
        sections = re.findall(r'section\s+(\d+(?:\.\d+)?)', query_lower)
        for section in sections:
            # Look for section mentions in content
            for i, chunk_meta in enumerate(self.chunk_metadata):
                content = chunk_meta['original_content'].lower()
                if f'section {section}' in content or f'{section}.' in content:
                    matched_indices.add(i)
        
        return list(matched_indices)

    def retrieve_relevant_chunks(self, query: str, k: int = 15) -> List[Tuple[str, float, str]]:
        """Much improved retrieval with multiple strategies"""
        try:
            if self.index is None:
                raise ValueError("Index not created. Call create_index() first.")
            
            logger.info(f"Retrieving content for query: '{query[:100]}...'")
            
            all_candidates = []
            seen_indices = set()
            
            # Strategy 1: Keyword-based exact matching (highest priority)
            keyword_matches = self._find_keyword_matches(query)
            for idx in keyword_matches:
                if idx < len(self.chunk_metadata):
                    metadata = self.chunk_metadata[idx]
                    content = metadata['original_content']
                    content_type = metadata['type']
                    
                    # High score for keyword matches
                    score = 0.9 + (0.1 * len(content.split()) / 100)  # Bonus for longer content
                    all_candidates.append((content, score, content_type, 'keyword'))
                    seen_indices.add(idx)
            
            logger.info(f"Found {len(keyword_matches)} keyword matches")
            
            # Strategy 2: Semantic search with very permissive threshold
            enhanced_query = query  # Use original query for better semantic matching
            query_embedding = self.embedding_model.encode([enhanced_query]).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Get many candidates
            search_k = min(k * 3, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, search_k)
            
            for score, idx in zip(scores[0], indices[0]):
                if idx >= len(self.chunk_metadata) or idx in seen_indices:
                    continue
                    
                metadata = self.chunk_metadata[idx]
                content = metadata['original_content']
                content_type = metadata['type']
                
                # Much more permissive threshold - accept almost everything
                if float(score) > -2.0:  # Very permissive
                    all_candidates.append((content, float(score), content_type, 'semantic'))
                    seen_indices.add(idx)
            
            # Strategy 3: Fuzzy content matching for missed items
            query_terms = set(re.findall(r'\b\w{4,}\b', query.lower()))
            
            for i, chunk_meta in enumerate(self.chunk_metadata):
                if i in seen_indices:
                    continue
                    
                content = chunk_meta['original_content'].lower()
                content_terms = set(re.findall(r'\b\w{4,}\b', content))
                
                # Calculate term overlap
                overlap = len(query_terms & content_terms)
                if overlap > 0:
                    overlap_score = overlap / max(len(query_terms), 1)
                    if overlap_score > 0.1:  # At least 10% term overlap
                        all_candidates.append((
                            chunk_meta['original_content'],
                            overlap_score * 0.5,  # Lower than semantic but still relevant
                            chunk_meta['type'],
                            'fuzzy'
                        ))
            
            # Sort by score (higher is better)
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Remove duplicates and take top k
            final_content = []
            seen_content_fingerprints = set()
            
            for content, score, ctype, method in all_candidates:
                # Create fingerprint based on first 200 characters
                fingerprint = content[:200].strip().lower()
                if fingerprint not in seen_content_fingerprints and len(final_content) < k:
                    final_content.append((content, score, ctype))
                    seen_content_fingerprints.add(fingerprint)
            
            # If still not enough results, add any remaining chunks with minimal scores
            if len(final_content) < 3:
                logger.warning("Very few results, adding more chunks with lower relevance")
                for i, chunk_meta in enumerate(self.chunk_metadata):
                    if len(final_content) >= k:
                        break
                    
                    content = chunk_meta['original_content']
                    fingerprint = content[:200].strip().lower()
                    
                    if fingerprint not in seen_content_fingerprints and len(content.strip()) > 50:
                        final_content.append((content, 0.1, chunk_meta['type']))
                        seen_content_fingerprints.add(fingerprint)
            
            logger.info(f"Retrieved {len(final_content)} relevant items")
            for i, (content, score, ctype) in enumerate(final_content[:5]):
                logger.info(f"  {i+1}. [{ctype}] Score: {score:.3f} - {content[:100]}...")
            
            return final_content
            
        except Exception as e:
            logger.error(f"Error retrieving content: {str(e)}")
            return []

    def generate_answer(self, query: str, relevant_content: List[Tuple[str, float, str]]) -> str:
        """Generate comprehensive answers using all available content"""
        try:
            if not relevant_content:
                return ("I couldn't find relevant information in the textbook to answer your question. "
                       "This could mean the content isn't available in the processed sections, or the question "
                       "requires information from parts that weren't successfully extracted. "
                       "Try asking about more specific topics or concepts.")
            
            # Organize content by relevance and type
            high_relevance = [item for item in relevant_content if item[1] > 0.5]
            medium_relevance = [item for item in relevant_content if 0.2 <= item[1] <= 0.5]
            low_relevance = [item for item in relevant_content if item[1] < 0.2]
            
            # Build comprehensive context
            context_parts = []
            
            if high_relevance:
                context_parts.append("=== HIGHLY RELEVANT CONTENT ===")
                for i, (content, score, ctype) in enumerate(high_relevance[:5], 1):
                    context_parts.append(f"[High Relevance {i}] ({ctype.upper()}, Score: {score:.2f})")
                    context_parts.append(content)
                    context_parts.append("")
            
            if medium_relevance:
                context_parts.append("=== MODERATELY RELEVANT CONTENT ===")
                for i, (content, score, ctype) in enumerate(medium_relevance[:5], 1):
                    context_parts.append(f"[Medium Relevance {i}] ({ctype.upper()}, Score: {score:.2f})")
                    context_parts.append(content)
                    context_parts.append("")
            
            if low_relevance and len(high_relevance) + len(medium_relevance) < 3:
                context_parts.append("=== ADDITIONAL CONTEXT ===")
                for i, (content, score, ctype) in enumerate(low_relevance[:3], 1):
                    context_parts.append(f"[Additional Context {i}] ({ctype.upper()}, Score: {score:.2f})")
                    context_parts.append(content)
                    context_parts.append("")
            
            context = "\n".join(context_parts)
            
            # Enhanced prompt for better responses
            prompt = f"""You are an expert mathematics tutor specializing in discrete mathematics. Your goal is to provide comprehensive, educational answers based on the textbook content provided.

IMPORTANT INSTRUCTIONS:
1. Use ALL the provided content to give a complete answer
2. If the question asks about a specific chapter/section, explain everything you can find about it
3. For broad topics, provide comprehensive overviews using all available information
4. Structure your answer clearly with headings and subpoints when appropriate
5. If you find definitions, theorems, or examples in the content, include them
6. Connect related concepts when you see them in different parts of the content
7. Be thorough - the student wants to learn as much as possible about the topic
8. If the content seems limited, work with what's available and be explicit about what you can explain

STUDENT'S QUESTION: {query}

AVAILABLE TEXTBOOK CONTENT:
{context}

Provide a comprehensive, well-structured educational response based on the available content:"""

            response = self.gemini_model.generate_content(prompt)
            
            if response.text:
                logger.info(f"Generated comprehensive answer for: '{query[:50]}...'")
                return response.text.strip()
            else:
                logger.warning("Gemini returned empty response")
                return "I couldn't generate a response. Please try rephrasing your question or asking about a more specific topic."
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I encountered an error while generating the answer: {str(e)}. Please try again with a more specific question."

    def answer_question(self, query: str, k: int = 15) -> Tuple[str, List[str], Dict[str, Any]]:
        """Main question answering method with improved retrieval and response"""
        try:
            logger.info(f"Processing question: '{query}'")
            
            # Retrieve with improved method
            relevant_content = self.retrieve_relevant_chunks(query, k)
            
            if not relevant_content:
                return (
                    "I couldn't find any relevant information for your question in the processed textbook content. "
                    "This could mean: 1) The topic isn't covered in the sections that were successfully processed, "
                    "2) The question needs to be more specific, or 3) There might be an issue with how the textbook was processed. "
                    "Try asking about specific mathematical concepts, theorems, definitions, or chapter numbers.",
                    [],
                    {}
                )
            
            # Generate comprehensive answer
            answer = self.generate_answer(query, relevant_content)
            
            # Prepare enhanced sources
            source_chunks = []
            visual_info = {"has_visual": False, "visual_count": 0, "visual_types": []}
            
            for content, score, content_type in relevant_content:
                preview = content[:400] + "..." if len(content) > 400 else content
                source_chunks.append(f"[{content_type.upper()}] (Relevance: {score:.2f}) {preview}")
                
                if content_type == "visual":
                    visual_info["has_visual"] = True
                    visual_info["visual_count"] += 1
                    
                    content_lower = content.lower()
                    if any(word in content_lower for word in ['equation', 'formula', 'mathematical']):
                        visual_info["visual_types"].append("Mathematical")
                    elif any(word in content_lower for word in ['diagram', 'figure', 'illustration']):
                        visual_info["visual_types"].append("Diagram")
                    elif any(word in content_lower for word in ['chart', 'graph', 'plot']):
                        visual_info["visual_types"].append("Chart")
                    else:
                        visual_info["visual_types"].append("Visual")
            
            logger.info(f"Generated answer with {len(source_chunks)} sources, {visual_info['visual_count']} visual elements")
            
            return answer, source_chunks, visual_info
            
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            return (
                f"I encountered an error processing your question: {str(e)}. "
                "Please try rephrasing your question or asking about a more specific topic.",
                [],
                {}
            )

    def get_visual_content_summary(self) -> Dict[str, Any]:
        """Get summary of visual content in the textbook"""
        if not self.visual_content:
            return {"total": 0, "by_type": {}, "by_page": {}}
        
        summary = {"total": len(self.visual_content), "by_type": {}, "by_page": {}}
        
        for visual in self.visual_content:
            vtype = visual.get('type', 'unknown')
            page = visual['page'] + 1
            
            summary["by_type"][vtype] = summary["by_type"].get(vtype, 0) + 1
            summary["by_page"][page] = summary["by_page"].get(page, 0) + 1
        
        return summary

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        if self.index is None:
            return {"status": "No index created"}
        
        return {
            "status": "Ready",
            "total_chunks": len(self.text_chunks),
            "visual_content": len(self.visual_content),
            "total_indexed": len(self.chunk_metadata),
            "embedding_dimension": self.embedding_dim,
            "index_size": self.index.ntotal,
            "keyword_terms": len(self.keyword_index),
            "chapters_indexed": len(self.chapter_index),
            "concepts_indexed": len(self.concept_index),
            "chunk_length_stats": {
                "avg_length": np.mean([meta["length"] for meta in self.chunk_metadata if meta["type"] == "text"]) if self.chunk_metadata else 0,
                "total_characters": sum([meta["length"] for meta in self.chunk_metadata if meta["type"] == "text"]) if self.chunk_metadata else 0
            }
        }