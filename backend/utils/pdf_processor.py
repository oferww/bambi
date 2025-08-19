import os
import PyPDF2
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, List, Optional
import tempfile
import shutil
from datetime import datetime
from .key_bank import get_keybank

# Optional fallbacks for PDF text extraction
try:
    import pdfplumber  # Better layout/word detection
except Exception:
    pdfplumber = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None

class PDFProcessor:
    """Process PDF files and extract text for RAG system."""
    
    def __init__(self, uploads_dir: str = "./data/uploads/pdfs", summary_length: str = "long"):
        """
        uploads_dir: directory to store uploaded PDFs
        summary_length: Cohere summarization length parameter ('short', 'medium', 'long', 'auto')
        """
        self.uploads_dir = uploads_dir
        os.makedirs(uploads_dir, exist_ok=True)

        # Initialize KeyBank; we'll create ChatCohere per call with best chat key
        self._keybank = get_keybank()
        self.summary_length = summary_length
        # Performance-tunable settings via env vars
        self.summary_mode = os.getenv("PDF_SUMMARY_MODE", "summarize").strip().lower()  # chat|summarize|off
        try:
            self.chunk_size = int(os.getenv("PDF_SUMMARY_CHUNK_SIZE", "4000"))
        except Exception:
            self.chunk_size = 4000
        try:
            self.chunk_overlap = int(os.getenv("PDF_SUMMARY_CHUNK_OVERLAP", "200"))
        except Exception:
            self.chunk_overlap = 200
        try:
            self.max_chars_cap = int(os.getenv("PDF_SUMMARY_MAX_CHARS", "120000"))  # cap work for very large docs
        except Exception:
            self.max_chars_cap = 120000
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text content from a PDF file with fallbacks and scanned detection."""
        # 1) Try PyPDF2 first
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_content.append(f"Page {page_num + 1}:\n{page_text.strip()}")
                if text_content:
                    return "\n\n".join(text_content)
        except Exception as e:
            print(f"[PDF] PyPDF2 extraction failed for {pdf_path}: {e}")

        # 2) Try pdfplumber for better layout handling and scanned detection
        if pdfplumber is not None:
            try:
                pages_text: List[str] = []
                scanned_pages = 0
                with pdfplumber.open(pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        words = []
                        try:
                            words = page.extract_words() or []
                        except Exception:
                            words = []
                        try:
                            ptxt = page.extract_text() or ""
                        except Exception:
                            ptxt = ""
                        if not ptxt.strip() and len(words) < 3:
                            scanned_pages += 1
                        if ptxt.strip():
                            pages_text.append(f"Page {i + 1}:\n{ptxt.strip()}")
                if scanned_pages > 0:
                    print(f"[PDF] Detected {scanned_pages} likely scanned page(s) in {os.path.basename(pdf_path)}. Consider enabling OCR.")
                if pages_text:
                    return "\n\n".join(pages_text)
            except Exception as e:
                print(f"[PDF] pdfplumber extraction failed for {pdf_path}: {e}")

        # 3) Try pdfminer.six as a final pure-text fallback
        if pdfminer_extract_text is not None:
            try:
                text = pdfminer_extract_text(pdf_path) or ""
                if text.strip():
                    # Add simple page markers if missing
                    return text
            except Exception as e:
                print(f"[PDF] pdfminer extraction failed for {pdf_path}: {e}")

        print(f"[PDF] Could not extract text from {pdf_path} with available methods.")
        return None
    
    def _split_long_text(self, text: str, max_chars: int = None, overlap: int = None) -> List[str]:
        """Split long input for summarization with overlap to preserve boundary context."""
        if max_chars is None:
            max_chars = self.chunk_size
        if overlap is None:
            overlap = self.chunk_overlap
        if len(text) <= max_chars:
            return [text]
        parts: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            slice_ = text[start:end]
            # prefer to break at paragraph boundary
            brk = slice_.rfind("\n\n")
            if brk == -1 or brk < max_chars * 0.5:
                brk = len(slice_)
            chunk = slice_[:brk]
            if chunk:
                parts.append(chunk)
            # move with overlap
            start = start + brk - overlap
            if start < 0:
                start = 0
            if start >= len(text):
                break
        return parts

    def _summarize_with_cohere(self, text: str) -> str:
        """Hierarchical summarization with chunk overlap using LangChain ChatCohere.
        """
        # Optionally cap total text to avoid excessive API calls
        capped_text = text[: self.max_chars_cap]
        if len(text) > self.max_chars_cap:
            print(f"[PDF] Truncated text to {self.max_chars_cap} chars for faster summarization", flush=True)

        # Allow skipping summarization entirely
        if self.summary_mode == "off":
            print("[PDF] Summarization disabled via PDF_SUMMARY_MODE=off (embedding raw text)", flush=True)
            return capped_text

        chunks = self._split_long_text(capped_text)
        model_choice = os.getenv("COHERE_CHAT_MODEL", os.getenv("PDF_SUMMARY_MODEL", "command-a-vision-07-2025")).strip()
        chunk_summaries: List[str] = []

        def _summarize_chunk_via_chat(chunk_text: str, idx: int) -> str:
            system_prompt = (
                "You are a faithful summarizer. Produce a concise but complete summary. "
                "Do not omit important facts, names, numbers, dates, or locations. "
                "Quote key phrases when necessary. If the input contains 'Page N:' markers, cite them."
            )
            msgs = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=(
                    "Summarize the following text faithfully. Preserve crucial details and include brief "
                    "citations to page markers if present.\n\n" + chunk_text
                )),
            ]
            # Acquire a fresh chat key from KeyBank and instantiate ChatCohere per call
            _chat_key = self._keybank.get_key("pdf_summary")
            chat = ChatCohere(
                model=os.getenv("COHERE_CHAT_MODEL", "command-a-vision-07-2025"),
                cohere_api_key=_chat_key,
                temperature=0.2,
                max_tokens=800,
            )
            resp = chat.invoke(msgs)
            return (getattr(resp, "content", None) or "").strip()

        # Legacy summarize API is removed to avoid direct SDK usage.
        def _summarize_chunk_via_summarize(chunk_text: str, length: str = "long") -> str:
            return _summarize_chunk_via_chat(chunk_text, 0)

        # Map step: summarize each chunk
        for i, ch in enumerate(chunks):
            summary = ""
            try:
                if model_choice.lower() in ("command-a-vision-07-2025", "command-r-plus", "command-r"):
                    # allow command-r as alternative
                    try:
                        summary = _summarize_chunk_via_chat(ch, i)
                    except Exception:
                        summary = _summarize_chunk_via_summarize(ch, self.summary_length)
                elif self.summary_mode == "summarize":
                    # Prefer summarize API (typically faster)
                    summary = _summarize_chunk_via_summarize(ch, self.summary_length)
                else:
                    # Auto: choose based on model_choice
                    if model_choice in ("command-r-plus", "command-r"):
                        try:
                            summary = _summarize_chunk_via_chat(ch, i)
                        except Exception:
                            summary = _summarize_chunk_via_summarize(ch, self.summary_length)
                    else:
                        summary = _summarize_chunk_via_summarize(ch, self.summary_length)
            except Exception as e:
                print(f"[PDF] Summarization error (chunk {i}): {e}")
                summary = ch[:1500]
            chunk_summaries.append(summary if summary else ch[:1500])

        combined = "\n\n".join(chunk_summaries)
        # Optional reduce step disabled by default for speed; enable via env
        if os.getenv("PDF_SUMMARY_ENABLE_REDUCE", "0") == "1" and len(combined) > 8000:
            try:
                if self.summary_mode in ("chat",) or model_choice in ("command-r-plus", "command-r"):
                    reduced = _summarize_chunk_via_chat(combined, -1)
                else:
                    reduced = _summarize_chunk_via_summarize(combined, "medium")
                combined = reduced or combined
            except Exception as e:
                print(f"[PDF] Final reduce summarization failed: {e}")
        return combined if combined else capped_text

    def process_pdf_file(self, pdf_file, filename: str) -> Dict[str, any]:
        """Process an uploaded PDF file and extract its content. Always save the PDF to uploads_dir."""
        save_path = os.path.join(self.uploads_dir, filename)
        try:
            # Save the uploaded file to the uploads_dir
            with open(save_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            # Extract text from the saved PDF
            raw_text = self.extract_text_from_pdf(save_path)
            
            if raw_text:
                # First summarize with Cohere, then embed chunks of the summary
                summary_text = self._summarize_with_cohere(raw_text)
                # Save summary alongside the PDF for user visibility
                try:
                    summary_path = f"{save_path}.summary.txt"
                    with open(summary_path, "w", encoding="utf-8") as sf:
                        sf.write(summary_text)
                    print(f"[PDF] Summary written to {summary_path}", flush=True)
                except Exception as werr:
                    print(f"[PDF] Could not write summary file for {filename}: {werr}", flush=True)

                return {
                    "filename": filename,
                    "content": summary_text,
                    "raw_text": raw_text,
                    "file_size": len(pdf_file.getbuffer()),
                    "pages": raw_text.count("Page ") + 1,
                    "type": "pdf_document"
                }
            else:
                return None
        except Exception as e:
            print(f"Error processing PDF file {filename}: {e}")
            return None
        # Do not delete the saved file; keep it for future reference
    
    def create_pdf_descriptions(self, pdf_data: Dict[str, any]) -> List[Dict[str, any]]:
        """Create descriptions for PDF content suitable for RAG.
        Returns both a summary document and a raw-text document so we embed both.
        """
        if not pdf_data or not pdf_data.get("content"):
            return []

        docs: List[Dict[str, any]] = []
        # Summary doc
        summary_description = f"Document: {pdf_data['filename']}\n\n{pdf_data['content']}"
        docs.append({
            "content": summary_description,
            "metadata": {
                "filename": pdf_data["filename"],
                "type": "pdf_document",
                "pages": pdf_data.get("pages", 1),
                "file_size": pdf_data.get("file_size", 0),
                "variant": "summary",
            }
        })

        # Raw doc (let the vectorstore splitter chunk it)
        if pdf_data.get("raw_text"):
            raw_description = f"Document (raw): {pdf_data['filename']}\n\n{pdf_data['raw_text']}"
            docs.append({
                "content": raw_description,
                "metadata": {
                    "filename": pdf_data["filename"],
                    "type": "pdf_document_raw",
                    "pages": pdf_data.get("pages", 1),
                    "file_size": pdf_data.get("file_size", 0),
                    "variant": "raw",
                }
            })

        return docs
