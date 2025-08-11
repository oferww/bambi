import os
import cohere
from typing import List, Dict, Any
from .rag_system import RAGSystem
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_community.llms import Cohere

class OferGPT:
    """Personal chatbot about Ofer using RAG with photos and memories."""
    
    def __init__(self):
        self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
        self.rag_system = RAGSystem()
        self.conversation_history = []  # Keep for backward compatibility
        
        # Initialize LangChain Cohere LLM for memory summarization
        self.langchain_llm = Cohere(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model=os.getenv("COHERE_CHAT_MODEL", "command-a-vision-07-2025"),
            temperature=0.35,  # Lower temperature for summarization
            max_tokens=200
        )
        
        # Initialize conversation memory with Cohere for smart summarization
        self.memory = ConversationSummaryBufferMemory(
            llm=self.langchain_llm,  # Pass Cohere LLM for summarization
            max_token_limit=800,  # Summarize when over 800 tokens
            return_messages=True,  # Return as message objects
            memory_key="chat_history"
        )
        
        # System prompt for the chatbot (friendlier/wittier, allow safe common-sense)
        self.system_prompt = """Your name is bambi. Your role is to be a personal AI assistant that knows about Ofer's life. you have information about his education, career, travels around the world, his cinema taste. 
\nCRITICAL: Ground your answers in the provided context. You may apply safe, widely-known common-sense inferences (e.g., mapping a city to its country), but do NOT invent new facts about Ofer that are not implied by context.

\n\nYour role is to:
\n1. Answer questions about Ofer's life and experiences based on provided context.
\n2. Keep tone friendly, concise, and a bit witty.
\n3. DO NOT fabricate personal details about ofer or events he was involved in.

\n\nWhen answering questions:
\n- Use information explicitly provided in the RAG context, and light, safe world knowledge (e.g., city->country).
\n- Be specific about dates, locations, and details ONLY when they are provided or safely implied.
\n- If you don't have information, say so honestly.
\n- DO NOT invent activities, meetings, or experiences not mentioned in the context.
\n- DO NOT reveal which RAG documents, files, or sources were used; do not list filenames, IDs, or metadata in answers.
\n- ALWAYS respond in English only; translate non-English names to English.
\n- Present dates in a natural format (e.g., 'May 13, 2010').

\n\nRemember: Be helpful and witty, but accurate."""
        
        # For debug: store last context and query
        self.last_rag_context = None
        self.last_user_query = None

    def _maybe_rerank(self, query: str, retrieved: list[dict]) -> list[dict]:
        """Optionally re-rank retrieved docs using Cohere Rerank.

        Env flags:
        - OFERGPT_RERANK: '1' => enable (default '0')
        - OFERGPT_RERANK_TOP_K: how many items from retrieved to re-rank (default from OFERGPT_RAG_TOP_K or 20)
        - OFERGPT_RERANK_MODEL: model name (default 'rerank-english-v3.0')

        Always falls back to the original list on any failure.
        """
        try:
            if os.getenv("OFERGPT_RERANK", "0") != "1":
                return retrieved
            if not retrieved:
                return retrieved
            try:
                rr_top_k = int(os.getenv("OFERGPT_RERANK_TOP_K", os.getenv("OFERGPT_RAG_TOP_K", "20")))
            except Exception:
                rr_top_k = 20

            head = retrieved[:rr_top_k]
            candidates = [d.get("content", "") or "" for d in head]
            # If all candidates are empty, nothing to do
            if not any(candidates):
                return retrieved

            rerank_model = os.getenv("OFERGPT_RERANK_MODEL", "rerank-english-v3.0")
            try:
                rr = self.cohere_client.rerank(
                    model=rerank_model,
                    query=query,
                    documents=candidates,
                    top_n=len(candidates),
                )
            except Exception:
                return retrieved

            results = getattr(rr, "results", None) or []
            if not results:
                return retrieved

            # Sort by relevance_score desc
            try:
                ordered = sorted(
                    [(r.index, getattr(r, "relevance_score", 0.0)) for r in results],
                    key=lambda x: x[1],
                    reverse=True,
                )
            except Exception:
                return retrieved

            # Reassemble: re-ordered head + untouched tail
            try:
                reordered_head = [head[idx] for idx, _ in ordered if 0 <= idx < len(head)]
                tail = retrieved[len(head):]
                return reordered_head + tail
            except Exception:
                return retrieved
        except Exception:
            return retrieved
    
    def _truncate(self, text: str, limit: int) -> str:
        """Truncate text to at most 'limit' characters, appending a notice if truncated."""
        try:
            limit = int(limit)
        except Exception:
            limit = 0
        if not text or limit <= 0:
            return "" if limit <= 0 else (text or "")
        if len(text) <= limit:
            return text
        # Leave room for suffix
        suffix = " … [TRUNCATED]"
        keep = max(0, limit - len(suffix))
        return text[:keep] + suffix

    def get_relevant_context(self, query: str) -> str:
        """Retrieve relevant context from the knowledge base for the query."""
        # Allow configuring how many documents to retrieve for RAG context
        try:
            top_k = int(os.getenv("OFERGPT_RAG_TOP_K", "20"))
        except Exception:
            top_k = 20
        # Budgets to avoid prompt overflow
        try:
            max_docs = int(os.getenv("OFERGPT_RAG_MAX_DOCS", "10"))
        except Exception:
            max_docs = 10
        try:
            ctx_budget = int(os.getenv("OFERGPT_RAG_CONTEXT_CHAR_BUDGET", "3500"))
        except Exception:
            ctx_budget = 3500
        try:
            per_doc_cap = int(os.getenv("OFERGPT_RAG_PER_DOC_CHAR_CAP", "900"))
        except Exception:
            per_doc_cap = 900
        retrieved = self.rag_system.search_similar(query, k=top_k)
        # Optional neural re-ranking (Cohere Rerank) to improve ordering
        retrieved = self._maybe_rerank(query, retrieved)
        # Wrap each document with clear markers and concise metadata header; apply per-doc cap
        doc_blocks = []
        for i, doc in enumerate(retrieved[:max_docs], start=1):
            meta = doc.get("metadata", {}) or {}
            t = str(meta.get("type") or meta.get("platform") or meta.get("source") or "document")
            date = (
                meta.get("date_only")
                or meta.get("date_rated")
                or meta.get("date_taken")
                or meta.get("timestamp")
                or meta.get("visit_date")
                or meta.get("date")
                or ""
            )
            loc = meta.get("location_name") or meta.get("country") or ""
            score = doc.get("similarity_score")
            try:
                score_str = f"{float(score):.3f}" if score is not None else ""
            except Exception:
                score_str = str(score) if score is not None else ""
            header_bits = []
            if t:
                header_bits.append(f"type={t}")
            if date:
                header_bits.append(f"date={date}")
            if loc:
                header_bits.append(f"loc={loc}")
            if score_str:
                header_bits.append(f"score={score_str}")
            header = "; ".join(header_bits)
            content = doc.get('content', '') or ''
            content = self._truncate(content, per_doc_cap)
            block = f"BEGIN DOC {i}{(' — ' + header) if header else ''}\n{content}\nEND DOC {i}"
            doc_blocks.append(block)
        # Assemble within global context budget
        context_parts = []
        remaining = ctx_budget
        for block in doc_blocks:
            if remaining <= 0:
                break
            sep = "\n\n" if context_parts else ""
            needed = len(sep) + len(block)
            if needed <= remaining:
                context_parts.append(sep + block if sep else block)
                remaining -= needed
            else:
                trunc_limit = max(0, remaining - len(sep))
                if trunc_limit > 0:
                    truncated = self._truncate(block, trunc_limit)
                    context_parts.append(sep + truncated if sep else truncated)
                    remaining = 0
                break
        context_str = "".join(context_parts)
        
        # Store for debug
        self.last_rag_context = context_str
        self.last_user_query = query
        return context_str

    # Cross-source feature removed
    
    # Location context removed
    
    def generate_response(self, query: str, context: str) -> str:
        prompt = f"""{self.system_prompt}

Context about Ofer:
{context}

User question: {query}

Please provide a helpful response based STRICTLY on the context about Ofer:"""
        try:
            response = self.cohere_client.generate(
                prompt=prompt,
                temperature=0.35,  # Lower temp for better instruction-following on retrieval tasks
                max_tokens=300,
                p=0.9,
                stop_sequences=["\n\nUser:", "\n\nQ:", "User question:", "Context about"]
            )
            answer = response.generations[0].text.strip()
            return answer
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble generating a response right now. Please try again."
    
    def generate_response_with_memory(self, query: str, context: str, memory_context: str) -> str:
        """Generate response using Cohere with RAG context and conversation memory."""
        try:
            # Truncate memory to stay within token limits
            try:
                mem_budget = int(os.getenv("OFERGPT_MEMORY_CHAR_BUDGET", "700"))
            except Exception:
                mem_budget = 700
            mem_text = memory_context if memory_context else "This is the start of our conversation."
            mem_text = self._truncate(mem_text, mem_budget)
            prompt = f"""{self.system_prompt}

Previous conversation summary:
{mem_text}

Current context about Ofer:
{context}

User question: {query}

Please provide a helpful response based STRICTLY on the context about Ofer and our previous conversation. DO NOT invent, assume, or make up any details not explicitly mentioned in the context above:"""
            response = self.cohere_client.generate(
                prompt=prompt,
                temperature=0.35,  # Lower temp for better instruction-following on retrieval tasks
                max_tokens=300,
                p=0.9,
                stop_sequences=["\n\nUser:", "\n\nQ:", "User question:", "Context about"]
            )
            answer = response.generations[0].text.strip()
            return answer
        except Exception as e:
            print(f"Error generating response with memory: {e}")
            return "I'm sorry, I'm having trouble generating a response right now. Please try again."
    
    def stream_response_with_memory(self, query: str, context: str, memory_context: str):
        """Generate streaming response using Cohere with RAG context and conversation memory."""
        import time
        try:
            print("🔄 Starting response generation...", flush=True)
            # Truncate memory to stay within token limits
            try:
                mem_budget = int(os.getenv("OFERGPT_MEMORY_CHAR_BUDGET", "700"))
            except Exception:
                mem_budget = 700
            mem_text = memory_context if memory_context else "This is the start of our conversation."
            mem_text = self._truncate(mem_text, mem_budget)
            prompt = f"""{self.system_prompt}

Previous conversation summary:
{mem_text}

Current context about Ofer:
{context}

Instruction: If a section labeled 'Cross-source' or enclosed by BEGIN/END CROSS-SOURCE appears above, prioritize it as primary evidence over other documents when answering. The context may contain multiple documents delimited by 'BEGIN DOC n' and 'END DOC n'. Treat each document as separate evidence; do not merge details across different documents unless they explicitly corroborate each other.

User question: {query}

Please provide a helpful response based STRICTLY on the context about Ofer and our previous conversation. DO NOT invent, assume, or make up any details not explicitly mentioned in the context above:"""
            try:
                print("🚀 Attempting real Cohere streaming...", flush=True)
                response = self.cohere_client.generate(
                    prompt=prompt,
                    temperature=0.35,  # Lower temp for better instruction-following on retrieval tasks
                    max_tokens=300,
                    p=0.9,
                    stop_sequences=["\n\nUser:", "\n\nQ:", "User question:", "Context about"],
                    stream=True
                )
                print("✅ Real Cohere streaming is working!", flush=True)
                full_response = ""
                token_count = 0
                for token in response:
                    text_segment = None
                    # Common Cohere stream shapes
                    if hasattr(token, 'generations') and getattr(token, 'generations'):
                        try:
                            text_segment = token.generations[0].text
                        except Exception:
                            text_segment = None
                    if text_segment is None and hasattr(token, 'text'):
                        text_segment = getattr(token, 'text')
                    # Some SDKs may yield dict-like events
                    if text_segment is None and isinstance(token, dict):
                        text_segment = token.get('text') or token.get('delta') or token.get('token')
                    if text_segment:
                        full_response += text_segment
                        token_count += 1
                        yield text_segment
                print(f"✅ Real streaming completed: {token_count} tokens received", flush=True)
                # If nothing streamed, force fallback
                if token_count == 0:
                    raise RuntimeError("Streaming returned 0 tokens; forcing fallback to non-streaming generation.")
            except Exception as stream_error:
                print(f"❌ Real streaming failed: {stream_error}", flush=True)
                print("🔄 Falling back to simulated streaming...", flush=True)
                response = self.cohere_client.generate(
                    prompt=prompt,
                    temperature=0.35,  # Lower temp for better instruction-following on retrieval tasks
                    max_tokens=300,
                    p=0.9,
                    stop_sequences=["\n\nUser:", "\n\nQ:", "User question:", "Context about"]
                )
                full_text = response.generations[0].text.strip()
                print(f"✅ Generated complete response: {len(full_text)} characters", flush=True)
                words = full_text.split(' ')
                print(f"🎭 Simulating streaming: {len(words)} words", flush=True)
                for i, word in enumerate(words):
                    if i == 0:
                        yield word
                    else:
                        yield ' ' + word
                    time.sleep(0.05)
                print("✅ Simulated streaming completed", flush=True)
        except Exception as e:
            print(f"Error generating streaming response: {e}", flush=True)
            yield "I'm sorry, I'm having trouble generating a response right now. Please try again."
    
    def chat(self, user_input: str) -> str:
        """Main chat method that combines RAG and memory."""
        # Get relevant context from RAG
        context = self.get_relevant_context(user_input)
        
        # Get conversation memory context
        memory_context = self.memory.buffer
        
        # Generate response with enhanced context
        response = self.generate_response_with_memory(user_input, context, memory_context)
        
        # Update both memory systems
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(response)
        
        # Keep backward compatibility with conversation_history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def chat_stream(self, user_input: str):
        """Main streaming chat method that combines RAG and memory."""
        # Get relevant context from RAG
        context = self.get_relevant_context(user_input)
        
        # Get conversation memory context
        memory_context = self.memory.buffer
        
        # Generate streaming response with enhanced context
        full_response = ""
        for token in self.stream_response_with_memory(user_input, context, memory_context):
            full_response += token
            yield token
        
        # Update both memory systems with the complete response
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(full_response)
        
        # Keep backward compatibility with conversation_history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": full_response})
    
    def add_photos_to_knowledge_base(self, photos_dir: str = "./data/uploads/photos"):
        """Add photos from directory to the knowledge base."""
        from .utils.photo_processor import PhotoProcessor
        
        processor = PhotoProcessor(photos_dir)
        photos_metadata = processor.process_photos_directory()
        photo_descriptions = processor.create_photo_descriptions(photos_metadata)
        
        self.rag_system.add_document_descriptions(photo_descriptions)
        
        return len(photos_metadata)
    
    def add_memories_to_knowledge_base(self, memories: List[str]):
        """Add text memories to the knowledge base."""
        self.rag_system.add_text_memories(memories)
    
    def add_pdf_to_knowledge_base(self, pdf_file, filename: str) -> bool:
        """Add a PDF document to the knowledge base."""
        try:
            from .utils.pdf_processor import PDFProcessor
            
            # Process the PDF file
            pdf_processor = PDFProcessor()
            pdf_data = pdf_processor.process_pdf_file(pdf_file, filename)
            
            if pdf_data:
                # Create descriptions for RAG
                pdf_descriptions = pdf_processor.create_pdf_descriptions(pdf_data)
                
                # Add to vector store
                self.rag_system.add_pdf_documents(pdf_descriptions)
                
                print(f"✅ PDF '{filename}' added to knowledge base", flush=True)
                return True
            else:
                print(f"❌ Failed to process PDF '{filename}'", flush=True)
                return False
                
        except Exception as e:
            print(f"❌ Error adding PDF to knowledge base: {e}", flush=True)
            return False
    
    def add_uploaded_photos_to_knowledge_base(self, uploaded_photos) -> int:
        """Add uploaded photos to the knowledge base."""
        try:
            from .utils.photo_processor import PhotoProcessor
            import os
            
            photos_dir = "./data/uploads/photos"
            os.makedirs(photos_dir, exist_ok=True)
            
            # Save uploaded photos to photos directory
            saved_photos = []
            for uploaded_photo in uploaded_photos:
                # Save the uploaded file to photos directory
                save_path = os.path.join(photos_dir, uploaded_photo.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_photo.getbuffer())
                saved_photos.append(save_path)
                print(f"Saved uploaded photo: {uploaded_photo.name}", flush=True)
            
            # Process the saved photos
            photo_processor = PhotoProcessor(photos_dir)
            photos_metadata = {}
            skipped_count = 0
            
            for photo_path in saved_photos:
                filename = os.path.basename(photo_path)
                metadata = photo_processor.extract_metadata(photo_path)
                if metadata is None:
                    print(f"❌ Skipping {filename}: location not found", flush=True)
                    skipped_count += 1
                    continue
                photos_metadata[filename] = metadata
            
            # Create photo descriptions
            photo_descriptions = photo_processor.create_photo_descriptions(photos_metadata)
            
            # Add to vector store
            self.rag_system.add_document_descriptions(photo_descriptions)
            
            processed_count = len(saved_photos)
            print(f"✅ Processed {processed_count} uploaded photos", flush=True)
            return processed_count
            
        except Exception as e:
            print(f"❌ Error adding uploaded photos to knowledge base: {e}", flush=True)
            return 0
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get information about the knowledge base."""
        return self.rag_system.get_collection_info()
    
    # Location cleaning removed
    
    def clear_knowledge_base(self):
        """Clear all data from the knowledge base."""
        self.rag_system.clear_vector_store()
        self.conversation_history = []
        self.memory.clear()  # Clear conversation memory too
    
    def clear_conversation_memory(self):
        """Clear only the conversation memory, keep knowledge base."""
        self.memory.clear()
        self.conversation_history = []
    
    # Location CSV generation removed
    
    # Location summary removed
    
    def fix_photo_location(self, filename: str, new_location_name: str, new_coordinates: str = ""):
        """Fix the location for a specific photo."""
        try:
            success = self.rag_system.update_photo_location(filename, new_location_name, new_coordinates)
            if success:
                print(f"✅ Fixed location for {filename}: {new_location_name}", flush=True)
                return True
            else:
                print(f"❌ Failed to fix location for {filename}", flush=True)
                return False
        except Exception as e:
            print(f"Error fixing photo location: {e}", flush=True)
            return False
    
    def fix_existing_photo_locations(self):
        """Fix all existing photos that have coordinates instead of location names."""
        try:
            self.rag_system.fix_existing_photo_locations()
            return True
        except Exception as e:
            print(f"Error fixing existing photo locations: {e}", flush=True)
            return False
