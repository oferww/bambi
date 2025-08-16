import os
import cohere
from typing import List, Dict, Any
import json
import concurrent.futures as cf
from .rag_system import RAGSystem
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_community.llms import Cohere

class OferGPT:
    """Personal chatbot about Ofer using RAG with photos and memories."""
    
    def __init__(self):
        self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY_CHAT"))
        self.rag_system = RAGSystem()
        self.conversation_history = []  # Keep for backward compatibility
        
        # Initialize LangChain Cohere LLM for memory summarization
        self.langchain_llm = Cohere(
            cohere_api_key=os.getenv("COHERE_API_KEY_CHAT"),
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
        self.system_prompt = """Your name is bambi. 
        Your role is to be a personal AI assistant that knows about Ofer's life. 
        you have information about his education, career, travels around the world, his cinema taste. 
        CRITICAL: Do NOT invent new facts about Ofer that are not implied by context. If you don't have information, say so honestly, ground your answers in the provided context.
        If you are not asked about Ofer, you don't have to answer about ofer, you have also information not related to Ofer, but you can use it if it helps you to answer the question.
        Dont mention documents names, dont mention documents content, dont mention documents source, dont mention documents metadata , dont mention documents number.
        Dont mention what you cant mention.
        You may apply safe, widely-known common-sense inferences e.g., mapping a city to its country. 
        Present dates in a natural format (e.g., 'May 13, 2010'). 
        Remember: Be helpful and witty, but accurate."""
        
        # For debug: store last context and query
        self.last_rag_context = None
        self.last_user_query = None

        # Intent categories the LLM may return when classifying user input
        self._intent_labels = {"greeting", "chitchat", "nonsense", "question", "ofer_question"}

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

    def _compute_relevant_context(self, query: str) -> str:
        """Compute the RAG context for a query (no timeout)."""
        # Allow configuring how many documents to retrieve for RAG context
        try:
            top_k = int(os.getenv("OFERGPT_RAG_TOP_K", "5"))
        except Exception:
            top_k = 5
        # Budgets to avoid prompt overflow
        try:
            max_docs = int(os.getenv("OFERGPT_RAG_MAX_DOCS", "5"))
        except Exception:
            max_docs = 5
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
        # Verbose logging of all RAG documents found (pre-cap)
        try:
            print(f"[RAG] Retrieved {len(retrieved)} documents:", flush=True)
            for i, doc in enumerate(retrieved, start=1):
                try:
                    meta = doc.get("metadata", {}) or {}
                    content = doc.get("content", "") or ""
                    # Prefer cosine similarity (higher is better) with raw distance for reference
                    cosine_sim = doc.get("cosine_similarity")
                    distance = doc.get("distance")
                    # Backward-compat: if only old field present, compute sim from it
                    if cosine_sim is None and doc.get("similarity_score") is not None:
                        try:
                            distance = float(doc.get("similarity_score"))
                            cosine_sim = 1.0 - distance
                        except Exception:
                            pass
                    meta_json = json.dumps(meta, ensure_ascii=False)
                    print(
                        f"[RAG][DOC {i}] cosine_similarity={cosine_sim} distance={distance}\n"
                        f"metadata={meta_json}\ncontent=\n{content}\n[END DOC {i}]\n",
                        flush=True,
                    )
                except Exception as _e:
                    print(f"[RAG] Failed to log doc {i}: {_e}", flush=True)
        except Exception as _e:
            print(f"[RAG] Verbose logging failed: {_e}", flush=True)
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
            # Prefer cosine similarity for header
            cosine_sim = doc.get("cosine_similarity")
            distance = doc.get("distance")
            # Backward-compat: derive from legacy field if needed
            if cosine_sim is None and doc.get("similarity_score") is not None:
                try:
                    distance = float(doc.get("similarity_score"))
                    cosine_sim = 1.0 - distance
                except Exception:
                    pass
            try:
                sim_str = f"{float(cosine_sim):.3f}" if cosine_sim is not None else ""
            except Exception:
                sim_str = str(cosine_sim) if cosine_sim is not None else ""
            header_bits = []
            if t:
                header_bits.append(f"type={t}")
            if date:
                header_bits.append(f"date={date}")
            if loc:
                header_bits.append(f"loc={loc}")
            if sim_str:
                header_bits.append(f"cos_sim={sim_str}")
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

    def get_relevant_context(self, query: str) -> str:
        """Retrieve relevant context with a timeout; if exceeded, return empty context.

        Timeout is configured by env var OFERGPT_RAG_TIMEOUT_SEC (default 20). Set to 0 to disable.
        """
        try:
            timeout_s = int(os.getenv("OFERGPT_RAG_TIMEOUT_SEC", "20"))
        except Exception:
            timeout_s = 20
        if timeout_s is not None and timeout_s > 0:
            try:
                with cf.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(self._compute_relevant_context, query)
                    return fut.result(timeout=timeout_s)
            except cf.TimeoutError:
                print(f"⏳ RAG timed out after {timeout_s}s; proceeding without documents.", flush=True)
                self.last_rag_context = ""
                self.last_user_query = query
                return ""
            except Exception as e:
                print(f"❌ RAG retrieval failed: {e}", flush=True)
                self.last_rag_context = ""
                self.last_user_query = query
                return ""
        # No timeout specified; compute directly
        return self._compute_relevant_context(query)
    
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
            if context and context.strip():
                prompt = f"{self.system_prompt}\n\nPrevious conversation summary:\n{mem_text}\n\nCurrent context about Ofer:\n{context}\n\nUser question: {query}\n\nPlease provide a helpful response based STRICTLY on the context about Ofer and our previous conversation. DO NOT invent, assume, or make up any details not explicitly mentioned in the context above:"
            else:
                # Smalltalk/casual mode (no RAG context)
                prompt = (
                    "You are bambi, a friendly, witty AI companion. The user is greeting or making small talk. "
                    "Reply naturally, concise but personable, vary phrasing (no canned lines), and optionally ask a light follow-up.\n\n"
                    f"Previous conversation summary:\n{mem_text}\n\n"
                    f"User message: {query}\n\n"
                    "Your response:"
                )
            # Log full Cohere API call parameters
            params = {
                "endpoint": "generate",
                "model": os.getenv("COHERE_CHAT_MODEL"),  # may be None for generate()
                "temperature": 0.35,
                "max_tokens": 300,
                "p": 0.9,
                "stop_sequences": ["\n\nUser:", "\n\nQ:", "User question:", "Context about"],
                "prompt_preview": prompt[:500]
            }
            print(f"[COHERE][REQUEST] {json.dumps(params, ensure_ascii=False)}", flush=True)
            # Simple timeout around Cohere call
            try:
                timeout_s = int(os.getenv("OFERGPT_COHERE_TIMEOUT_SEC", "25"))
            except Exception:
                timeout_s = 20
            if timeout_s and timeout_s > 0:
                with cf.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(
                        self.cohere_client.generate,
                        prompt=prompt,
                        temperature=0.35,  # Lower temp for better instruction-following on retrieval tasks
                        max_tokens=300,
                        p=0.9,
                        stop_sequences=["\n\nUser:", "\n\nQ:", "User question:", "Context about"]
                    )
                    response = fut.result(timeout=timeout_s)
            else:
                response = self.cohere_client.generate(
                    prompt=prompt,
                    temperature=0.35,  # Lower temp for better instruction-following on retrieval tasks
                    max_tokens=300,
                    p=0.9,
                    stop_sequences=["\n\nUser:", "\n\nQ:", "User question:", "Context about"]
                )
            # Log full response object (best-effort)
            try:
                print(f"[COHERE][RESPONSE] {response}", flush=True)
            except Exception:
                try:
                    print(f"[COHERE][RESPONSE_DICT] {getattr(response, '__dict__', {})}", flush=True)
                except Exception:
                    pass
            answer = response.generations[0].text.strip()
            return answer
        except cf.TimeoutError:
            print("⏳ Cohere generate() timed out; returning fallback message.", flush=True)
            return "I'm sorry, the request took too long. Please try again."
        except Exception as e:
            print(f"Error generating response with memory: {e}")
            return "I'm sorry, I'm having trouble generating a response right now. Please try again."
    
    def _detect_intent_llm(self, query: str, memory_context: str) -> str:
        """Use the LLM to classify high-level intent.

        Returns one of: greeting, chitchat, nonsense, question, ofer_question.
        Falls back to 'question' on any failure.
        """
        try:
            instruction = (
                "Classify the user's message into exactly one of these intents: "
                "greeting, chitchat, nonsense, question, ofer_question. "
                "Definitions: greeting = hello/hi/etc; chitchat = casual small talk; "
                "nonsense = empty or meaningless; ofer_question = specifically about Ofer; "
                "question = general question; ofer_question = specifically about Ofer. "
                "Respond ONLY with the intent word, no punctuation, no explanation."
            )
            mem_text = memory_context or ""
            prompt = (
                f"Instruction: {instruction}\n\n"
                f"Conversation summary (may be empty):\n{mem_text}\n\n"
                f"User message:\n{query}\n\n"
                "Intent:"
            )
            try:
                timeout_s = int(os.getenv("OFERGPT_INTENT_TIMEOUT_SEC", "5"))
            except Exception:
                timeout_s = 5
            # Safe logging without preview variable
            try:
                _prompt_len = len(prompt)
            except Exception:
                _prompt_len = -1
            print(
                f"[INTENT][REQUEST] timeout={timeout_s}s prompt_len={_prompt_len} max_tokens=8",
                flush=True,
            )
            # Use Cohere Chat API (Generate is deprecated)
            chat_kwargs = {
                "message": prompt,
                "temperature": 0.0,
                "max_tokens": 8,
            }
            _chat_model = os.getenv("COHERE_CHAT_MODEL")
            if _chat_model:
                chat_kwargs["model"] = _chat_model

            if timeout_s and timeout_s > 0:
                with cf.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(self.cohere_client.chat, **chat_kwargs)
                    resp = fut.result(timeout=timeout_s)
            else:
                resp = self.cohere_client.chat(**chat_kwargs)

            raw_intent = (getattr(resp, "text", None) or "").strip()
            print(f"[INTENT][RAW_RESPONSE] {raw_intent!r}", flush=True)
            intent = raw_intent.lower()
            if intent in self._intent_labels:
                print(f"[INTENT][DECISION] intent={intent} (matched known labels)", flush=True)
                return intent
            print(f"[INTENT][DECISION] intent_unrecognized={intent!r} -> fallback='question'", flush=True)
            return "question"
        except Exception as e:
            print(f"[INTENT] fallback due to error: {e}", flush=True)
            return "question"

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
            if context and context.strip():
                prompt = (
                    f"{self.system_prompt}\n\n"
                    f"Previous conversation summary:\n{mem_text}\n\n"
                    f"Current context about Ofer:\n{context}\n\n"
                    f"User question: {query}\n\n"
                    "Please provide a helpful response based STRICTLY on the context about Ofer and our previous conversation. "
                    "DO NOT invent, assume, or make up any details not explicitly mentioned in the context above:"
                )
            else:
                # Smalltalk/casual mode (no RAG context)
                prompt = (
                    "You are bambi, a friendly, witty AI companion. The user is greeting or making small talk. "
                    "Reply naturally and briefly, keep it dynamic.\n\n"
                    f"Previous conversation summary:\n{mem_text}\n\n"
                    f"User message: {query}\n\n"
                    "Your response:"
                )
            try:
                print("🚀 Attempting real Cohere streaming...", flush=True)
                # Log full Cohere API call parameters for streaming
                params = {
                    "endpoint": "generate",
                    "model": os.getenv("COHERE_CHAT_MODEL"),  # may be None for generate()
                    "temperature": 0.35,
                    "max_tokens": 300,
                    "p": 0.9,
                    "stop_sequences": ["\n\nUser:", "\n\nQ:", "User question:", "Context about"],
                    "stream": True,
                    "prompt_preview": prompt[:500]
                }
                print(f"[COHERE][REQUEST] {json.dumps(params, ensure_ascii=False)}", flush=True)
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
        """Main chat method that combines intent routing, memory, and RAG."""
        # Normalize input
        user_input = user_input.strip()
        # Pull memory first for intent detection context
        memory_context = self.memory.buffer
        # LLM-based intent detection to optionally skip RAG
        intent = self._detect_intent_llm(user_input, memory_context)
        if intent in {"greeting", "chitchat", "nonsense"}:
            # Smalltalk path: no RAG, call generator with empty context
            response = self.generate_response_with_memory(user_input, "", memory_context)
        else:
            # Information-seeking: run RAG
            context = self.get_relevant_context(user_input)
            response = self.generate_response_with_memory(user_input, context, memory_context)

        # Update both memory systems
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(response)
        
        # Keep backward compatibility with conversation_history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def chat_stream(self, user_input: str):
        """Main streaming chat method with LLM-based intent routing to skip RAG for greetings/smalltalk."""
        user_input = user_input.strip()
        memory_context = self.memory.buffer
        intent = self._detect_intent_llm(user_input, memory_context)
        full_response = ""
        if intent in {"greeting", "chitchat", "nonsense"}:
            # Smalltalk path: no RAG, stream with empty context
            for token in self.stream_response_with_memory(user_input, "", memory_context):
                full_response += token
                yield token
        else:
            # Information-seeking: run RAG
            context = self.get_relevant_context(user_input)
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
    
    def clear_knowledge_base(self):
        """Clear all data from the knowledge base."""
        self.rag_system.clear_vector_store()
        self.conversation_history = []
        self.memory.clear()  # Clear conversation memory too
    
    def clear_conversation_memory(self):
        """Clear only the conversation memory, keep knowledge base."""
        self.memory.clear()
        self.conversation_history = []
