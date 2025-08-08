import fitz  # PyMuPDF
import re
import logging
from typing import List, Tuple, Optional, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Significantly improved PDF processor with better text extraction and preservation"""
    
    def __init__(self):
        # More conservative chunking to preserve context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # Larger chunks for better context
            chunk_overlap=200,  # More overlap to prevent context loss
            length_function=len,
            separators=[
                "\n\n\n",  # Chapter/section breaks
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentence endings
                "! ",
                "? ",
                "; ",
                ", ",
                " ",       # Word boundaries
                ""         # Character level
            ]
        )

    def extract_text_and_images(self, pdf_path: str) -> Tuple[str, Optional[str], List[Dict[str, Any]]]:
        """Extract text and images with much better content preservation"""
        try:
            doc = fitz.open(pdf_path)
            text_content = ""
            title = None
            visual_content = []
            
            logger.info(f"Processing PDF with {len(doc)} pages")
            
            # Try to get title from metadata
            metadata = doc.metadata
            if metadata and metadata.get('title'):
                title = metadata['title'].strip()
                logger.info(f"Found title in metadata: {title}")
            
            # Process each page with much better text extraction
            for page_num in range(len(doc)):
                page = doc[page_num]
                logger.debug(f"Processing page {page_num + 1}")
                
                try:
                    # Extract text with better preservation
                    page_text = page.get_text()
                    
                    if page_text and page_text.strip():
                        # Much more conservative cleaning to preserve content
                        cleaned_text = self._preserve_content_cleaning(page_text, page_num)
                        if cleaned_text.strip():
                            text_content += cleaned_text + "\n\n"
                            logger.debug(f"Extracted {len(cleaned_text)} characters from page {page_num + 1}")
                    
                    # Try to extract title from first few pages if not found
                    if not title and page_num < 3 and page_text:
                        potential_title = self._extract_title_from_text(page_text)
                        if potential_title:
                            title = potential_title
                            logger.info(f"Extracted title from page {page_num + 1}: {title}")
                    
                except Exception as text_error:
                    logger.error(f"Error extracting text from page {page_num + 1}: {str(text_error)}")
                
                # Extract visual content
                try:
                    page_visual_content = self._extract_visual_content_from_page(page, page_num)
                    visual_content.extend(page_visual_content)
                except Exception as visual_error:
                    logger.warning(f"Error extracting visual content from page {page_num + 1}: {str(visual_error)}")
            
            doc.close()
            
            # Final content improvement without losing information
            text_content = self._final_content_preservation(text_content)
            
            logger.info(f"Extraction complete:")
            logger.info(f" - Text content: {len(text_content)} characters")
            logger.info(f" - Visual content: {len(visual_content)} items")
            logger.info(f" - Title: {title or 'Not found'}")
            
            # More lenient content check
            if len(text_content.strip()) < 50 and len(visual_content) == 0:
                logger.error("Very little content extracted - file might be corrupted or image-based")
                raise ValueError("Could not extract sufficient content from PDF")
            
            return text_content, title, visual_content
            
        except Exception as e:
            logger.error(f"Error extracting content from PDF: {str(e)}")
            raise e

    def _preserve_content_cleaning(self, text: str, page_num: int) -> str:
        """Content-preserving cleaning that keeps almost everything"""
        if not text or not text.strip():
            return ""
        
        # Remove only obvious artifacts, keep everything else
        text = text.replace('\x00', '').replace('\ufffd', '')
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip only completely empty lines
            if not line:
                continue
            
            # Skip only obvious page numbers that are alone on a line
            if re.match(r'^\d{1,3}$', line) and int(line) <= 1000:
                continue
                
            # Skip only clear header/footer patterns
            if re.match(r'^Page \d+$', line, re.IGNORECASE):
                continue
                
            # Keep everything else - be very conservative about removal
            # Fix obvious text issues but don't remove content
            line = self._minimal_text_fixes(line)
            
            # Smart line joining for broken sentences
            if (cleaned_lines and 
                cleaned_lines[-1] and 
                not cleaned_lines[-1].endswith(('.', '!', '?', ':', ';', '-')) and
                line and 
                line[0].islower() and
                not line.startswith(('•', '-', '◦', '(', '['))):
                # Merge with previous line
                cleaned_lines[-1] += ' ' + line
            else:
                cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        return result

    def _minimal_text_fixes(self, line: str) -> str:
        """Minimal fixes that don't remove content"""
        # Fix spacing issues
        line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)  # Missing space between words
        line = re.sub(r'([.!?])([A-Z])', r'\1 \2', line)  # Missing space after punctuation
        line = re.sub(r' {2,}', ' ', line)  # Multiple spaces
        
        return line.strip()

    def _final_content_preservation(self, text: str) -> str:
        """Final cleanup that preserves as much content as possible"""
        if not text:
            return ""
        
        # Very gentle normalization
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # Limit excessive line breaks
        text = re.sub(r' {3,}', '  ', text)       # Limit excessive spaces
        
        # Preserve paragraph structure
        lines = text.split('\n')
        final_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Keep all non-empty lines
                final_lines.append(line)
            elif final_lines and final_lines[-1]:  # Preserve paragraph breaks
                final_lines.append('')
        
        result = '\n'.join(final_lines)
        
        # Final gentle formatting
        result = re.sub(r'\n\s*\n\s*\n\s*\n', '\n\n\n', result)  # Max 3 line breaks
        
        return result

    def chunk_text(self, text: str) -> List[str]:
        """Improved text chunking that preserves more content and context"""
        if not text or not text.strip():
            logger.warning("No text content to chunk")
            return []
        
        try:
            logger.info(f"Chunking text of {len(text)} characters")
            
            # First pass: try with our configured splitter
            chunks = self.text_splitter.split_text(text)
            
            # Much more permissive filtering
            filtered_chunks = []
            for chunk in chunks:
                chunk = chunk.strip()
                
                # Keep chunks with minimal requirements
                if len(chunk) > 30:  # Very low threshold
                    # Improve chunk quality without losing content
                    improved_chunk = self._improve_chunk_preserving_content(chunk)
                    
                    # Check if chunk has any meaningful content
                    if self._chunk_has_content(improved_chunk):
                        filtered_chunks.append(improved_chunk)
            
            logger.info(f"Created {len(filtered_chunks)} chunks from {len(text)} characters")
            
            # If we lost too much content, try smaller chunks
            if len(filtered_chunks) < 3 and len(text) > 500:
                logger.info("Few chunks created, trying with smaller chunk size...")
                smaller_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600,
                    chunk_overlap=100,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                chunks = smaller_splitter.split_text(text)
                filtered_chunks = []
                
                for chunk in chunks:
                    chunk = chunk.strip()
                    if len(chunk) > 20:  # Even more permissive
                        improved_chunk = self._improve_chunk_preserving_content(chunk)
                        if self._chunk_has_content(improved_chunk):
                            filtered_chunks.append(improved_chunk)
                
                logger.info(f"Created {len(filtered_chunks)} smaller chunks")
            
            # Ultimate fallback: paragraph-based chunking
            if len(filtered_chunks) < 2 and len(text.strip()) > 100:
                logger.warning("Using paragraph-based fallback chunking")
                paragraphs = text.split('\n\n')
                filtered_chunks = []
                
                current_chunk = ""
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    if len(current_chunk) + len(para) < 1500:
                        current_chunk += para + "\n\n"
                    else:
                        if current_chunk.strip():
                            filtered_chunks.append(current_chunk.strip())
                        current_chunk = para + "\n\n"
                
                if current_chunk.strip():
                    filtered_chunks.append(current_chunk.strip())
                
                logger.info(f"Fallback created {len(filtered_chunks)} paragraph-based chunks")
            
            # Absolute fallback: return the whole text
            if not filtered_chunks and len(text.strip()) > 50:
                logger.warning("No chunking worked, returning full text as single chunk")
                filtered_chunks = [text.strip()]
            
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            # Emergency fallback
            if len(text.strip()) > 50:
                return [text.strip()]
            return []

    def _improve_chunk_preserving_content(self, chunk: str) -> str:
        """Improve chunk quality while preserving all content"""
        # Just normalize whitespace, don't remove anything
        chunk = re.sub(r'\n+', ' ', chunk)  # Replace line breaks with spaces
        chunk = re.sub(r' {2,}', ' ', chunk)  # Fix multiple spaces
        chunk = chunk.strip()
        
        return chunk

    def _chunk_has_content(self, chunk: str) -> bool:
        """Check if chunk has meaningful content - very permissive"""
        if not chunk or len(chunk.strip()) < 10:
            return False
        
        # Check for actual words
        words = re.findall(r'\b\w+\b', chunk)
        if len(words) < 3:
            return False
        
        # Avoid chunks that are just numbers or symbols
        if re.match(r'^[\d\s\W]+, chunk'):
            return False
        
        return True

    def _extract_visual_content_from_page(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract and analyze visual content from a page"""
        visual_content = []
        
        try:
            # Extract images
            image_list = page.get_images(full=True)
            logger.debug(f"Found {len(image_list)} images on page {page_num + 1}")
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    visual_item = {
                        'page': page_num,
                        'index': img_index,
                        'width': base_image.get("width", 0),
                        'height': base_image.get("height", 0),
                        'format': image_ext,
                        'size_bytes': len(image_bytes),
                        'type': self._classify_visual_content_type(image_bytes, image_ext),
                        'description': f"Visual content on page {page_num + 1}"
                    }
                    
                    # Try to generate better description
                    try:
                        description = self._analyze_visual_content(image_bytes, image_ext, page_num)
                        if description:
                            visual_item['description'] = description
                    except Exception:
                        pass
                    
                    visual_content.append(visual_item)
                    
                except Exception as img_error:
                    logger.warning(f"Could not process image {img_index} from page {page_num + 1}: {str(img_error)}")
                    continue
            
            # Look for mathematical expressions in text
            try:
                page_text = page.get_text()
                math_expressions = self._extract_mathematical_content(page_text, page_num)
                visual_content.extend(math_expressions)
            except Exception:
                pass
                
        except Exception as e:
            logger.warning(f"Error extracting visual content from page {page_num + 1}: {str(e)}")
        
        return visual_content

    def _classify_visual_content_type(self, image_bytes: bytes, image_ext: str) -> str:
        """Classify the type of visual content"""
        size = len(image_bytes)
        if size < 5000:
            return "mathematical"
        elif size > 100000:
            return "photograph"
        else:
            return "diagram"

    def _analyze_visual_content(self, image_bytes: bytes, image_ext: str, page_num: int) -> str:
        """Analyze visual content and generate description"""
        try:
            description = self._try_ocr_on_image(image_bytes, image_ext)
            if description and len(description.strip()) > 10:
                return f"[Visual Content - Page {page_num + 1}] Text found in image: {description[:200]}..."
        except:
            pass
        
        return f"[Visual Content - Page {page_num + 1}] Image or diagram"

    def _try_ocr_on_image(self, image_bytes: bytes, image_ext: str) -> Optional[str]:
        """Try to extract text from image using OCR"""
        try:
            from PIL import Image
            import pytesseract
            import io
            
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            if text and text.strip():
                return text.strip()
        except ImportError:
            logger.debug("OCR libraries not available (PIL/pytesseract)")
        except Exception:
            pass
        
        return None

    def _extract_mathematical_content(self, page_text: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract mathematical expressions from text"""
        math_expressions = []
        
        # Enhanced mathematical patterns
        math_patterns = [
            r'[A-Za-z]\s*[=]\s*[^,\n]{1,50}',  # Variable equations
            r'[A-Za-z]²\s*[\+\-]\s*[A-Za-z]²',  # Quadratic expressions
            r'\d+\s*[\+\-\*\/]\s*\d+\s*[=]\s*\d+',  # Arithmetic
            r'[∑∏∫∪∩⊆⊇]',  # Mathematical symbols
            r'\b[A-Za-z]+\([A-Za-z,\s]+\)\s*=',  # Function definitions
            r'Theorem\s+\d+', r'Definition\s+\d+', r'Lemma\s+\d+',  # Mathematical statements
        ]
        
        for i, pattern in enumerate(math_patterns):
            matches = re.finditer(pattern, page_text, re.IGNORECASE)
            for match in matches:
                math_expressions.append({
                    'page': page_num,
                    'index': i,
                    'width': 0,
                    'height': 0,
                    'format': 'text',
                    'size_bytes': len(match.group().encode()),
                    'type': 'mathematical',
                    'description': f"[Mathematical Content - Page {page_num + 1}] {match.group()}"
                })
        
        return math_expressions

    def _extract_title_from_text(self, text: str) -> Optional[str]:
        """Try to extract title from text content"""
        if not text:
            return None
        
        lines = text.split('\n')
        for line in lines[:15]:  # Check first 15 lines
            line = line.strip()
            
            # Look for title-like patterns
            if (10 < len(line) < 100 and
                not line.lower().startswith(('chapter', 'section', 'page', 'table')) and
                not re.search(r'\d{4}', line) and  # Avoid years/dates
                line.count(' ') > 1):  # Multi-word
                
                # Check if it's likely a title
                words = line.split()
                if len(words) >= 2:
                    # Check for title case or all caps
                    title_case = sum(1 for w in words if w[0].isupper()) >= len(words) // 2
                    if title_case or line.isupper():
                        return line
        
        return None

    # Legacy and utility methods
    def extract_text(self, pdf_path: str) -> Tuple[str, Optional[str]]:
        """Extract text from PDF file (legacy method)"""
        text, title, _ = self.extract_text_and_images(pdf_path)
        return text, title

    def extract_table_of_contents(self, pdf_path: str) -> List[Dict[str, any]]:
        """Extract table of contents if available"""
        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()
            doc.close()
            
            if not toc:
                return []
            
            toc_entries = []
            for level, title, page in toc:
                toc_entries.append({
                    'level': level,
                    'title': title.strip(),
                    'page': page
                })
            
            logger.info(f"Extracted {len(toc_entries)} TOC entries")
            return toc_entries
            
        except Exception as e:
            logger.warning(f"Could not extract TOC: {str(e)}")
            return []

    def extract_page_text(self, pdf_path: str, page_num: int) -> str:
        """Extract text from a specific page"""
        try:
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                raise ValueError(f"Page {page_num} does not exist")
            
            page = doc[page_num]
            text = page.get_text()
            doc.close()
            
            return self._preserve_content_cleaning(text, page_num)
            
        except Exception as e:
            logger.error(f"Error extracting page {page_num}: {str(e)}")
            raise e

    def save_images_to_disk(self, visual_content: List[Dict], output_dir: str = "extracted_images") -> List[str]:
        """Save extracted visual content to disk"""
        import os
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            saved_files = []
            
            for i, item in enumerate(visual_content):
                if 'data' in item:  # Only save actual image data
                    filename = f"page_{item['page']}_img_{item['index']}.{item.get('format', 'png')}"
                    filepath = os.path.join(output_dir, filename)
                    
                    with open(filepath, "wb") as f:
                        f.write(item['data'])
                    
                    saved_files.append(filepath)
                    logger.info(f"Saved visual content: {filepath}")
            
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving visual content: {str(e)}")
            raise e