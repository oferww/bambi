import streamlit as st
import os
from dotenv import load_dotenv
from backend.chatbot import OferGPT
from backend.utils.photo_processor import PhotoProcessor
# Auto-create location embeddings on startup
from backend.rag_system import RAGSystem
from backend.utils.s3_sync import sync_s3_prefix_to_dir, sync_dir_to_s3_prefix
import json
from datetime import datetime
import pandas as pd
from backend.ingestion import (
    ingest_files,
    ingest_scan_uploads,
    ingest_pdfs_in_uploads,
    ingest_csvs_in_uploads,
    ingest_instagram_jsons_in_uploads,
    ingest_photos_from_photos_dir,
)
import random
import html as html_lib
# Load environment variables
load_dotenv()

# Sidebar visibility control (OFERGPT_HIDE_SIDEBAR=1 to hide)
HIDE_SIDEBAR = os.getenv("OFERGPT_HIDE_SIDEBAR", "0") == "1"

# Page configuration
# Resolve assets relative to this file (frontend/)
BASE_DIR = os.path.dirname(__file__)

_icon_candidates = [
    os.path.join(BASE_DIR, "assets", "icon.png")
]
_page_icon = next((p for p in _icon_candidates if os.path.exists(p)), "🦌")
st.set_page_config(
    page_title="bambi",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state=("collapsed" if HIDE_SIDEBAR else "expanded"),
)

# Rotating chat input placeholders
PLACEHOLDERS = [
    "Ask me about Ofer's life... 🦌",
    "Ask me about Ofer's education... 📚",
    "Ask me about Ofer's career... 🏆",
    "Ask me about Ofer's travels... 🌍",
    "Ask me about Ofer's cinema taste... 🎬",
    "Spill the Ofer tea… ☕",
    "What’s the latest on Planet Ofer? 🪐",
    "Ask me something Ofer-the-top 😏",
    "Career gossip about Ofer? 🧑‍💼",
    "Got Ofer lore to unlock? 📖",
    "Where in the world is Ofer now? 🌍",
    "Tell me your Ofer thoughts, I’ll rank them. 🏆",
    "Ofer’s origin story, act 1? 🎬",
    "What did Ofer learn this time? 🎓",
    "Plot twist in Ofer’s saga? 🔀",
    "Which Ofer timeline are we in? ⏳",
    "Travel tales: Ofer edition ✈️",
    "Hot takes on Ofer’s movies? 🍿",
    "If Ofer were a meme…? 😂",
    "Two truths and a lie about Ofer 🤫",
    "What’s Ofer cooking today? 🍳",
    "Ask me like I’m Ofer’s diary 📓",
    "Break the Ofer-ice 🧊",
    "Pitch Ofer’s biopic title 🎥",
    "One Ofer fact to rule them all 💍",
    "Plot Ofer on a map 🗺️",
    "Ofer’s playlist vibes? 🎧",
    "Give me an Ofer riddle 🧩",
    "What would Ofer do? (WWOD) 🧠",
]

# Witty, on-theme dynamic thinking lines
THINKING_PHRASES = {
    "Running through the meadow, searching for the answer" : "🦌",
    "Leafing through Ofer’s memories… quite literally" : "🍃",
    "Sniffing out Ofer facts on the trail" : "🦌",
    "Tracking footprints from Ofer’s travels" : "✈️",
    "Browsing Ofer’s film shelf for clues" : "🎬",
    "Studying Ofer’s syllabus like midterms" : "📚",
    "Hiking across Ofer’s career path for hints" : "🏞️",
    "Mapping Ofer’s story one breadcrumb at a time" : "🗺️",
    "Tuning into Ofer’s soundtrack for signals" : "🎧",
    "Dusting off Ofer lore in the archives" : "🗄️",
    "Following a green trail to the truth" : "🌿",
    "Cross-referencing Ofer’s chapters… page by page" : "📖",
}

if "current_placeholder" not in st.session_state:
    st.session_state.current_placeholder = random.choice(PLACEHOLDERS)

def _next_placeholder():
    """Pick a different placeholder for the next prompt."""
    options = [p for p in PLACEHOLDERS if p != st.session_state.current_placeholder]
    if options:
        st.session_state.current_placeholder = random.choice(options)
    # If options is empty (single item list), keep as is

@st.cache_resource
def run_s3_sync_once() -> bool:
    """Run S3 pull"""
    try:
        S3_BUCKET = os.getenv("S3_BUCKET")
        if not S3_BUCKET:
            print("[S3] Skipping: S3_BUCKET not configured", flush=True)
            return False

        S3_PREFIX = os.getenv("S3_PREFIX", "")
        S3_REGION = os.getenv("S3_REGION")
        AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
        # Prefer S3_ENDPOINT; fall back to legacy S3_ENDPOINT_URL
        S3_ENDPOINT = os.getenv("S3_ENDPOINT") or os.getenv("S3_ENDPOINT_URL")

        # Normalize base prefix once
        base_prefix = (S3_PREFIX or "").strip().strip("/")

        # Always fetch embeddings on demand
        embeddings_dir = "./data/embeddings"
        os.makedirs(embeddings_dir, exist_ok=True)
        embeddings_prefix = f"{base_prefix}/embeddings" if base_prefix else "embeddings"
        print(f"[S3] Pulling embeddings: s3://{S3_BUCKET}/{embeddings_prefix} -> {embeddings_dir}", flush=True)
        sync_s3_prefix_to_dir(
            bucket=S3_BUCKET,
            prefix=embeddings_prefix,
            local_dir=embeddings_dir,
            region=S3_REGION,
            access_key=AWS_ACCESS_KEY_ID,
            secret_key=AWS_SECRET_ACCESS_KEY,
            session_token=AWS_SESSION_TOKEN,
            endpoint_url=S3_ENDPOINT,
            overwrite=False,
        )

        return True
    except Exception as e:
        print(f"[S3] Skipping S3 pull due to error: {e}", flush=True)
        return False

# Optional one-time full sync on server start (controlled by a single flag)
if os.getenv("OFERGPT_S3_SYNC_ON_START", "0") == "1":
    run_s3_sync_once()

# Explicit favicon link (PNG)
_fav_links = []
_png = os.path.join(BASE_DIR, "assets", "icon.png")

# Manual on-demand sync button in sidebar
if st.sidebar.button("Sync from S3"):
    ok = run_s3_sync_once()
    if ok:
        st.sidebar.success("S3 sync completed.")
    else:
        st.sidebar.warning("S3 sync skipped or failed. Check logs.")

st.markdown("""
<div class="top-banner">
  Powered by Streamlit, LangChain, Cohere, and ChromaDB. <br>
  Source code available at <a href="https://github.com/oferww/bambi">github.com/oferww/bambi</a>
</div>
""", unsafe_allow_html=True)

# If hiding sidebar, inject CSS to remove it from the layout entirely
if HIDE_SIDEBAR:
    st.markdown(
        """
        <style>
          [data-testid="stSidebar"] { display: none !important; }
          /* Also hide the sidebar hamburger if present */
          button[kind="header"] { display: none !important; }
          /* Hide the collapsed sidebar arrow/control */
          [data-testid="collapsedControl"] { display: none !important; visibility: hidden !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Hide Streamlit top bar, header, and footer
st.markdown(
    """
    <style>
      /* Hide the Streamlit top toolbar/hamburger */
      div[data-testid="stToolbar"] { display: none !important; }
      #MainMenu { visibility: hidden; }

      /* Hide the header entirely */
      header[data-testid="stHeader"], [data-testid="stHeader"] {
        display: none !important;
      }

      /* Hide the footer */
      footer, [data-testid="stFooter"] { display: none !important; visibility: hidden !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Smoother background and readable content container
st.markdown(
    """
    <style>
      html, body { background: #e8f5e9 !important; }
      .stApp {
        /* Single light green background */
        background: #e8f5e9 !important; /* light green 50 */
        background-attachment: fixed !important;
      }

      /* Make all content areas transparent so green shows everywhere */
      .block-container {
        background: transparent !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        box-shadow: none !important;
        padding-top: 6px !important; /* bring content closer to top */
      }

      /* Also clear common inner wrappers to avoid white strips */
      [data-testid="stAppViewContainer"],
      [data-testid="stMain"] {
        background: transparent !important;
      }

      /* Make chat and sidebar surfaces transparent as well */
      [data-testid="stChatMessage"],
      [data-testid="stChatMessageContent"],
      [data-testid="stChatMessageAvatar"],
      [data-testid="stChatInput"],
      [data-testid="stSidebar"],
      .st-emotion-cache-1wmy9hl, /* common container class fallback */
      .st-emotion-cache-1kyxreq {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
      }

      /* Transparent header */
      [data-testid="stHeader"] { background: transparent !important; }

      /* Top banner (scrolls with content, not sticky) */
      .top-banner {
        margin: 0;
        padding: 4px 0;
        text-align: center;
        color: #666;
        font-size: 0.9rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Centered header logo (no fullscreen) and hide fullscreen UI button on images
st.markdown(
    """
    <style>
      .header-logo { display:flex; justify-content:center; align-items:center; margin: 0.75rem 0 0.25rem; }
      .header-logo img { height: 72px; width: auto; }
      /* Hide Streamlit image fullscreen button */
      button[title="View fullscreen"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Render static header logo once at top-level (outside main)
# Ensures the logo is present immediately and not tied to first prompt re-run
_header_logo = os.path.join(BASE_DIR, "assets", "bambi.png")
if os.path.exists(_header_logo):
    import base64
    with open(_header_logo, "rb") as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode("utf-8")
    st.markdown(
        f'''
        <div style="display:flex; justify-content:center; align-items:center; margin: 10px 0 6px 0;">
            <img src="data:image/png;base64,{_logo_b64}" alt="bambi" style="width:340px; height:auto;" />
        </div>
        ''',
        unsafe_allow_html=True,
    )

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .assistant-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    /* Override Streamlit's native chat message colors to keep everything blue */
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: #e3f2fd !important;
    }
    .stChatMessage[data-testid="chat-message-assistant"] .stMarkdown {
        color: #1565c0 !important;
    }
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: #e3f2fd !important;
    }
    .stChatMessage[data-testid="chat-message-user"] .stMarkdown {
        color: #1565c0 !important;
    }
    
    /* Ensure streaming content stays blue */
    .stChatMessage .stMarkdown p {
        color: #1565c0 !important;
    }
    
    /* Loading indicator animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-spinner {
        animation: spin 1s linear infinite;
        display: inline-block;
        font-size: 1.2em;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced chat theming (non-breaking, CSS-only)
st.markdown(
    """
    <style>
      /* Constrain content width for better readability */
      .block-container { max-width: 1100px; margin: auto; }

      /* Smooth scrollbars */
      ::-webkit-scrollbar { width: 10px; height: 10px; }
      ::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 6px; }
      ::-webkit-scrollbar-track { background: transparent; }

      /* Chat message bubble styling */
      [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.45) !important; /* subtle glass over green */
        border: 1px solid rgba(0,0,0,0.06) !important;
        border-radius: 16px !important;
        padding: 12px 14px !important;
        margin: 10px 0 !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06) !important;
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
      }

      /* Left notch for assistant, right notch for user */
      [data-testid="stChatMessage"][data-testid="chat-message-assistant"]::after,
      [data-testid="stChatMessage"][data-testid="chat-message-user"]::after { content: ""; position: absolute; width: 0; height: 0; }

      /* Avatars: subtle ring and consistent size; keep contain for assistant image */
      [data-testid="stChatMessageAvatar"] img {
        width: 36px !important; height: 36px !important; object-fit: contain !important; border-radius: 10px !important;
        background: #fff !important; box-shadow: 0 2px 8px rgba(0,0,0,0.12) !important;
      }

      /* Right-align user bubble while preserving your custom layout */
      [data-testid="stChatMessage"][data-testid="chat-message-user"] {
        border-left: none !important;
        border-right: 3px solid #ff9800 !important; /* orange accent */
      }

      /* Assistant bubble accent */
      [data-testid="stChatMessage"][data-testid="chat-message-assistant"] {
        border-left: 3px solid #2e7d32 !important; /* deep green accent */
      }

      /* Markdown content polish */
      [data-testid="stChatMessageContent"] code {
        background: rgba(0,0,0,0.06) !important; padding: 2px 6px; border-radius: 6px; font-size: 0.95em;
      }
      [data-testid="stChatMessageContent"] pre code {
        display: block; padding: 14px; border-radius: 12px; line-height: 1.4;
        background: #0b1020; color: #e5f3ff; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08);
      }
      [data-testid="stChatMessageContent"] h1, 
      [data-testid="stChatMessageContent"] h2, 
      [data-testid="stChatMessageContent"] h3 {
        margin-top: 0.2rem; color: #1b5e20; /* deep green headings */
      }

      /* Subtle message entrance animation */
      @keyframes messageIn { from { transform: translateY(6px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
      [data-testid="stChatMessage"] { animation: messageIn 220ms ease-out; }

      /* Chat input polish */
      [data-testid="stChatInput"] textarea {
        background: rgba(255,255,255,0.78) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.06) !important;
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        padding: 12px 14px !important;
        font-size: 1rem !important;
        line-height: 1.4 !important;
        caret-color: #2e7d32 !important; /* deep green caret */
        transition: box-shadow 120ms ease, border-color 120ms ease;
      }
      [data-testid="stChatInput"] textarea::placeholder { color: rgba(0,0,0,0.35); }
      [data-testid="stChatInput"] textarea:focus {
        outline: none !important;
        border-color: rgba(46,125,50,0.45) !important;
        box-shadow: 0 0 0 3px rgba(76,175,80,0.20) !important; /* green glow */
      }


      /* Timestamp (if present in message text via small tag) */
      [data-testid="stChatMessageContent"] small { color: rgba(0,0,0,0.45); }

      /* Responsive tweaks */
      @media (max-width: 768px) {
        .block-container { padding-left: 0.6rem; padding-right: 0.6rem; }
        [data-testid="stChatMessageAvatar"] img { width: 32px !important; height: 32px !important; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'knowledge_base_initialized' not in st.session_state:
    st.session_state.knowledge_base_initialized = False
if 'pending_response' not in st.session_state:
    st.session_state.pending_response = False


def initialize_chatbot():
    """Initialize the chatbot with Cohere API key."""
    api_key = os.getenv("COHERE_API_KEY_CHAT")
    if not api_key:
        st.error("❌ COHERE_API_KEY_CHAT not found in environment variables!")
        st.info("Please set your Cohere API key in the .env file")
        return None
    
    try:
        chatbot = OferGPT()
        st.session_state.chatbot = chatbot
        return chatbot
    except Exception as e:
        st.error(f"❌ Error initializing chatbot: {e}")
        return None

def add_photos_to_knowledge_base():
    """Add photos from the photos directory to the knowledge base."""
    if not st.session_state.chatbot:
        st.error("Chatbot not initialized!")
        return
    
    photos_dir = "./data/uploads/photos"
    if not os.path.exists(photos_dir):
        st.warning(f"Photos directory not found: {photos_dir}")
        return
    
    photos = [f for f in os.listdir(photos_dir) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
    
    if not photos:
        st.warning("No photos found in the photos directory!")
        return
    
    with st.spinner(f"Processing {len(photos)} photos..."):
        try:
            count = st.session_state.chatbot.add_photos_to_knowledge_base(photos_dir)
            st.success(f"✅ Successfully processed {count} photos!")
            st.session_state.knowledge_base_initialized = True
        except Exception as e:
            st.error(f"❌ Error processing photos: {e}")

def add_memory():
    """Add a text memory to the knowledge base."""
    if not st.session_state.chatbot:
        st.error("Chatbot not initialized!")
        return
    
    memory_text = st.text_area("Enter your memory:", height=100)
    if st.button("Add Memory"):
        if memory_text.strip():
            try:
                st.session_state.chatbot.add_memories_to_knowledge_base([memory_text])
                st.success("✅ Memory added successfully!")
                st.session_state.knowledge_base_initialized = True
            except Exception as e:
                st.error(f"❌ Error adding memory: {e}")
        else:
            st.warning("Please enter some text for the memory.")

# Removed display_chat_message function - now using st.chat_message consistently

# Main application
def main():
    # Auto-initialize chatbot on app start
    if not st.session_state.get('chatbot'):
        with st.spinner("🚀 Initializing bambi..."):
            chatbot = initialize_chatbot()

    
    # Sidebar
    if not HIDE_SIDEBAR:
        with st.sidebar:
            st.markdown("## ⚙️ Settings")
        
            # Show last RAG context for debugging
            if st.button("Show Last RAG Context"):
                if st.session_state.chatbot:
                    last_query = getattr(st.session_state.chatbot, 'last_user_query', None)
                    last_context = getattr(st.session_state.chatbot, 'last_rag_context', None)
                    with st.expander("Last RAG Context and Query", expanded=True):
                        st.markdown(f"**Last User Query:**\n\n{last_query if last_query else 'N/A'}")
                        st.markdown("---")
                        st.markdown(f"**Last RAG Context:**\n\n")
                        st.code(last_context if last_context else 'N/A', language='markdown')
            
            # Knowledge base info
            st.markdown("#### 📊 Knowledge Base Info")
            info = st.session_state.chatbot.get_knowledge_base_info()
            if "error" not in info:
                # Show document counts side-by-side
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total Docs", info.get("total_docs", 0))
                col2.metric("Photos", info.get("num_photos", 0))
                col3.metric("PDFs", info.get("num_pdfs", 0))
                col4.metric("Memories", info.get("num_memories", 0))
                col5.metric("Locations", info.get("num_locations", 0))
                st.caption("Docs = unique photos, PDFs, memories, locations (not chunked)")
                st.metric("Total RAG Chunks", info.get("total_rag_chunks", 0))
                st.caption("Total number of vector store chunks (used for retrieval)")
                st.metric("Collection", info.get("collection_name", "N/A"))
                st.metric("Model", info.get("embedding_model", "N/A"))

                # Live Country Stats removed
            else:
                st.error(f"Error: {info['error']}")
            # Show chatbot status
            if st.session_state.get('chatbot'):
                st.success("✅ Chatbot ready!")
        
            # Knowledge base management
            st.markdown("### 📚 Knowledge Base")

            if st.session_state.chatbot:
                # Unified Add Files (images, PDFs, CSVs)
                st.markdown("#### 📁 Add Files")
                uploaded_files = st.file_uploader(
                    "Drop photos, PDF documents, or CSVs:",
                    type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'pdf', 'csv'],
                    accept_multiple_files=True,
                    help="Upload images, PDFs, or CSV files to add to Ofer's knowledge base"
                )

                if uploaded_files:
                    if st.button("🚀 Process Files"):
                        if not st.session_state.chatbot:
                            st.error("Please initialize the chatbot first!")
                        else:
                            with st.spinner("Processing files…"):
                                try:
                                    photos_added, pdfs_added, csvs_processed = ingest_files(uploaded_files, st.session_state.chatbot.rag_system)
                                    msgs = []
                                    if photos_added:
                                        msgs.append(f"📸 {photos_added} photos")
                                    if pdfs_added:
                                        msgs.append(f"📄 {pdfs_added} PDFs")
                                    if csvs_processed:
                                        msgs.append(f"🗺️ {csvs_processed} CSVs")
                                    if msgs:
                                        st.success("✅ Processed: " + ", ".join(msgs))
                                    else:
                                        st.info("No new documents were added (likely duplicates)")
                                except Exception as e:
                                    st.error(f"❌ Error processing files: {e}")
                    else:
                        with st.spinner("Processing files…"):
                            try:
                                photos_added, pdfs_added, csvs_processed = ingest_files(uploaded_files, st.session_state.chatbot.rag_system)
                                msgs = []
                                if photos_added:
                                    msgs.append(f"📸 {photos_added} photos")
                                if pdfs_added:
                                    msgs.append(f"📄 {pdfs_added} PDFs")
                                if csvs_processed:
                                    msgs.append(f"🗺️ {csvs_processed} CSVs")
                                if msgs:
                                    st.success("✅ Processed: " + ", ".join(msgs))
                                else:
                                    st.info("No new documents were added (likely duplicates)")
                            except Exception as e:
                                st.error(f"❌ Error processing files: {e}")

            
            # Scan existing uploads folder
            st.markdown("#### 📂 Scan uploads folders")
            st.caption("Process files already in subfolders: ./data/uploads/json, ./data/uploads/pdfs, ./data/uploads/csv, ./data/uploads/photos")
            if st.button("🔎 Scan & Ingest from ./data/uploads/*"):
                with st.spinner("Scanning and ingesting uploads …"):
                    try:
                        photos_added, pdfs_added, csvs_processed = ingest_scan_uploads(st.session_state.chatbot.rag_system)
                        msgs = []
                        if photos_added:
                            msgs.append(f"📸 {photos_added} posts/photos")
                        if pdfs_added:
                            msgs.append(f"📄 {pdfs_added} PDFs")
                        if csvs_processed:
                            msgs.append(f"🗺️ {csvs_processed} CSVs")
                        if msgs:
                            st.success("✅ Ingested: " + ", ".join(msgs))
                        else:
                            st.info("No new documents were added (likely duplicates)")
                    except Exception as e:
                        st.error(f"❌ Error scanning uploads: {e}")

            # Manual per-type embedding controls
            st.markdown("#### 🧩 Embed by type")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("📸 Embed Photos"):
                    with st.spinner("Embedding photos from ./data/uploads/photos …"):
                        try:
                            count = ingest_photos_from_photos_dir(st.session_state.chatbot.rag_system, photos_dir="./data/uploads/photos")
                            if count:
                                st.success(f"✅ Embedded {count} new photos")
                            else:
                                st.info("No new photos (likely duplicates)")
                        except Exception as e:
                            st.error(f"❌ Error embedding photos: {e}")
            with col_b:
                if st.button("🗺️ Embed CSVs"):
                    with st.spinner("Embedding CSVs from ./data/uploads/csv …"):
                        try:
                            count = ingest_csvs_in_uploads(st.session_state.chatbot.rag_system)
                            if count:
                                st.success(f"✅ Embedded {count} CSV(s)")
                            else:
                                st.info("No CSVs found or already embedded")
                        except Exception as e:
                            st.error(f"❌ Error embedding CSVs: {e}")

            col_c, col_d = st.columns(2)
            with col_c:
                if st.button("📄 Embed PDFs"):
                    with st.spinner("Embedding PDFs from ./data/uploads/pdfs …"):
                        try:
                            count = ingest_pdfs_in_uploads(st.session_state.chatbot.rag_system)
                            if count:
                                st.success(f"✅ Embedded {count} PDF(s)")
                            else:
                                st.info("No new PDFs (likely duplicates)")
                        except Exception as e:
                            st.error(f"❌ Error embedding PDFs: {e}")
            with col_d:
                if st.button("🧾 Embed JSONs"):
                    with st.spinner("Embedding Instagram posts_*.json from ./data/uploads/json …"):
                        try:
                            count = ingest_instagram_jsons_in_uploads(st.session_state.chatbot.rag_system)
                            if count:
                                st.success(f"✅ Embedded {count} post(s)")
                            else:
                                st.info("No new posts (likely duplicates)")
                        except Exception as e:
                            st.error(f"❌ Error embedding JSONs: {e}")

            # PDF Summaries viewer
            st.markdown("#### 📝 PDF Summaries")
            try:
                uploads_dir = "./data/uploads/pdfs"
                if os.path.exists(uploads_dir):
                    # Collect .summary.txt files and sort by modified time (newest first)
                    all_files = [
                        (f, os.path.getmtime(os.path.join(uploads_dir, f)))
                        for f in os.listdir(uploads_dir)
                        if f.lower().endswith(".summary.txt")
                    ]
                    all_files.sort(key=lambda x: x[1], reverse=True)
                    if not all_files:
                        st.caption("No PDF summaries found yet. Upload a PDF to generate one.")
                    for fname, _ in all_files[:50]:  # cap to 50 entries for UI
                        fpath = os.path.join(uploads_dir, fname)
                        try:
                            with open(fpath, 'r', encoding='utf-8') as fh:
                                content = fh.read()
                        except Exception as rerr:
                            content = f"<error reading summary: {rerr}>"
                        # Derive original PDF name by stripping '.summary.txt'
                        pdf_name = fname[:-12] if fname.lower().endswith('.summary.txt') else fname
                        with st.expander(f"📄 {pdf_name}"):
                            st.text_area("Summary", content, height=200, label_visibility="collapsed")
                            st.download_button(
                                label="📥 Download Summary",
                                data=content,
                                file_name=fname,
                                mime="text/plain",
                                use_container_width=True
                            )
                else:
                    st.caption("Uploads folder not found yet. It will be created on first PDF upload.")
            except Exception as e:
                st.error(f"Error loading PDF summaries: {e}")

            # Instagram posts viewer (direct from vector store)
            st.markdown("#### 📸 Instagram Posts")
            try:
                collection = st.session_state.chatbot.rag_system.chroma_client.get_or_create_collection("ofergpt_memories")
                data = collection.get()
                posts = []
                if data and data.get("metadatas"):
                    import ast as _ast
                    aggregated = {}
                    docs = data.get("documents", []) or []
                    metas = data.get("metadatas", []) or []
                    for idx, (content, meta) in enumerate(zip(docs, metas)):
                        if not meta:
                            continue
                        # Accept any Instagram marker (platform/source/type)
                        _plat = str(meta.get("platform") or meta.get("source") or meta.get("type") or "").lower()
                        if _plat != "instagram":
                            continue
                        # Stable key per post across chunks
                        key = meta.get("instagram_id") or meta.get("idempotency_key") or meta.get("filename") or str(idx)

                        # Start with meta-level fields for this chunk
                        cap = meta.get("caption", "")
                        location_name = meta.get("location_name") or meta.get("location") or ""
                        coords = meta.get("coordinates")

                        # Parse content JSON if present
                        if content and isinstance(content, str) and content.strip().startswith("{"):
                            try:
                                cobj = json.loads(content)
                                cap = cobj.get("caption", cap)
                                coords = coords or cobj.get("coordinates")
                                location_name = location_name or cobj.get("location_name", "")
                            except Exception:
                                pass

                        # Normalize coordinates for this chunk (handle dict or string serialized dict)
                        lat = None
                        lon = None
                        if isinstance(coords, str):
                            s = coords.strip()
                            # Try JSON first then Python literal if string looks like a dict
                            if s.startswith("{") and s.endswith("}"):
                                try:
                                    try:
                                        coords_obj = json.loads(s)
                                    except Exception:
                                        import ast as __ast
                                        coords_obj = __ast.literal_eval(s)
                                    if isinstance(coords_obj, dict):
                                        coords = coords_obj
                                except Exception:
                                    pass
                            # Regex fallback for strings that aren't valid JSON/literal
                            if not isinstance(coords, dict):
                                try:
                                    import re as __re
                                    mlat = __re.search(r"lat[^\d-]*(-?\d+(?:\.\d+)?)", s, flags=__re.IGNORECASE)
                                    mlon = __re.search(r"lon[gitude]*[^\d-]*(-?\d+(?:\.\d+)?)", s, flags=__re.IGNORECASE)
                                    if mlat and mlon:
                                        coords = {"lat": float(mlat.group(1)), "lon": float(mlon.group(1))}
                                except Exception:
                                    pass
                        if isinstance(coords, dict):
                            lat = coords.get("lat", lat)
                            lon = coords.get("lon", lon)
                        # Final meta-level fallback if still missing
                        if (lat is None or lon is None):
                            mlat = meta.get("latitude") or meta.get("lat")
                            mlon = meta.get("longitude") or meta.get("lon")
                            try:
                                if lat is None and mlat is not None:
                                    lat = float(mlat) if isinstance(mlat, (int, float, str)) and str(mlat).strip() else lat
                                if lon is None and mlon is not None:
                                    lon = float(mlon) if isinstance(mlon, (int, float, str)) and str(mlon).strip() else lon
                            except Exception:
                                pass

                        # Merge fields per post key (field-wise), avoiding overwriting non-empty values
                        cur = aggregated.get(key, {
                            "caption": "",
                            "timestamp": "",
                            "location": "",
                            "latitude": "",
                            "longitude": "",
                        })
                        if (not cur.get("caption")) and cap:
                            cur["caption"] = cap
                        if (not cur.get("location")) and location_name:
                            cur["location"] = location_name
                        if (not cur.get("timestamp")) and meta.get("timestamp"):
                            cur["timestamp"] = meta.get("timestamp", "")
                        if cur.get("latitude", "") == "" and lat is not None:
                            cur["latitude"] = lat
                        if cur.get("longitude", "") == "" and lon is not None:
                            cur["longitude"] = lon
                        aggregated[key] = cur

                    # Drop helper keys and materialize rows
                    posts = []
                    for v in aggregated.values():
                        posts.append({
                            "caption": v.get("caption", ""),
                            "timestamp": v.get("timestamp", ""),
                            "location": v.get("location", ""),
                            "latitude": v.get("latitude", ""),
                            "longitude": v.get("longitude", ""),
                        })
                # Filter UI
                q = st.text_input("Search caption/location", value="", placeholder="type to filter…")
                if q:
                    ql = q.lower()
                    posts = [p for p in posts if ql in (p.get("caption", "") + " " + p.get("location", "")).lower()]
                st.caption(f"Showing {len(posts)} post(s)")
                if posts:
                    df = pd.DataFrame(posts)
                    # Normalize dtypes for Arrow compatibility and clean display
                    try:
                        for col in ("caption", "timestamp", "location"):
                            if col in df.columns:
                                df[col] = df[col].astype(str)
                        def _fmt_coord(v):
                            try:
                                if v is None or (isinstance(v, str) and not v.strip()):
                                    return ""
                                return f"{float(v):.6f}"
                            except Exception:
                                return str(v)
                        if "latitude" in df.columns:
                            df["latitude"] = df["latitude"].apply(_fmt_coord)
                        if "longitude" in df.columns:
                            df["longitude"] = df["longitude"].apply(_fmt_coord)
                    except Exception:
                        pass
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No Instagram posts found yet. Use '🧾 Embed JSONs' or 'Scan uploads' after placing posts_*.json in ./data/uploads.")

                # # Maintenance actions
                # col_ri, col_fix, col_rf = st.columns(3)
                # with col_ri:
                #     if st.button("♻️ Reingest Instagram (delete & import)"):
                #         with st.spinner("Deleting existing Instagram docs and re-ingesting …"):
                #             try:
                #                 rag = st.session_state.chatbot.rag_system
                #                 deleted = rag.delete_by_filter({"platform": "instagram"})
                #                 added = ingest_instagram_jsons_in_uploads(rag)
                #                 # After ingest, resolve locations from coordinates
                #                 fixed = rag.fix_existing_instagram_locations()
                #                 st.success(f"✅ Deleted {deleted}, added {added} and resolved {fixed} Instagram post(s)")
                #                 st.rerun()
                #             except Exception as e:
                #                 st.error(f"❌ Reingest failed: {e}")
                # with col_fix:
                #     if st.button("🧭 Resolve Instagram locations (from coordinates)"):
                #         try:
                #             with st.spinner("Resolving Instagram locations…"):
                #                 rag = st.session_state.chatbot.rag_system
                #                 fixed = rag.fix_existing_instagram_locations()
                #                 st.success(f"Resolved {fixed} Instagram post(s)")
                #                 st.rerun()
                #         except Exception as e:
                #             st.error(f"Failed to resolve locations: {e}")
                with col_rf:
                    if st.button("🔄 Refresh Instagram list"):
                        st.rerun()
            except Exception as e:
                st.error(f"Error loading Instagram posts: {e}")

            # Add memories
            st.markdown("#### 💭 Add Memories")
            add_memory()
            
            # Location Tracking and CSV removed
            
            # Memory Management
            st.markdown("#### 🧠 Memory Management")
            
            # Show memory status
            try:
                memory_messages = len(st.session_state.chatbot.memory.chat_memory.messages)
                st.metric("Conversation Memory", f"{memory_messages} messages")
            except:
                st.metric("Conversation Memory", "0 messages")
            
            # Clear conversation memory only
            if st.button("🧹 Clear Conversation Memory"):
                st.session_state.chatbot.clear_conversation_memory()
                st.session_state.chat_history = []  # Clear Streamlit chat history too
                st.success("✅ Conversation memory cleared!")
                st.rerun()
            
            # Non-English location cleaning removed
            
            # Clear knowledge base (also clears memory)
            if st.button("🗑️ Clear Knowledge Base", type="secondary"):
                st.session_state.chatbot.clear_knowledge_base()
                st.session_state.knowledge_base_initialized = False
                st.session_state.chat_history = []  # Clear Streamlit chat history too
                st.success("Knowledge base and memory cleared!")
                st.rerun()
            
            # Fix All Photo Locations removed
        
            # Show memory summary for debugging
            if st.button("Show Memory Summary"):
                if st.session_state.chatbot:
                    memory_summary = getattr(st.session_state.chatbot.memory, 'buffer', None)
                    max_token_limit = getattr(st.session_state.chatbot.memory, 'max_token_limit', None)
                    with st.expander("Conversation Memory Summary", expanded=True):
                        st.markdown(f"**Memory Window (max tokens):** {max_token_limit if max_token_limit else 'N/A'}")
                        st.markdown("---")
                        st.markdown(f"**Current Memory Summary:**\n\n")
                        st.code(memory_summary if memory_summary else 'N/A', language='markdown')
            
            # Debug photos with location data
            if st.button("🔍 Debug Photo Locations"):
                if st.session_state.chatbot:
                    try:
                        # Get all documents from vector store
                        collection = st.session_state.chatbot.rag_system.chroma_client.get_or_create_collection("ofergpt_memories")
                        result = collection.get()
                        
                        st.markdown("#### 📸 Photos in Vector Store:")
                        
                        for i, (doc_id, metadata) in enumerate(zip(result['ids'], result['metadatas'])):
                            if metadata and metadata.get('filename'):
                                filename = metadata.get('filename', 'Unknown')
                                date_taken = metadata.get('date_taken', 'Unknown')
                                location_name = metadata.get('location_name', 'No location')
                                coordinates = metadata.get('coordinates', 'No coordinates')
                                
                                with st.expander(f"📷 {filename} ({date_taken})"):
                                    st.write(f"**Location:** {location_name}")
                                    st.write(f"**Coordinates:** {coordinates}")
                                    st.write(f"**Document ID:** {doc_id}")
                                    
                                    # Show full metadata
                                    st.json(metadata)
                                    
                    except Exception as e:
                        st.error(f"Error debugging photos: {e}")
            

    # Main chat interface
    # Chat input (fixed footer, outside columns to satisfy Streamlit constraints)
    user_input = st.chat_input(placeholder=st.session_state.get("current_placeholder", "Ask me about Ofer's life..."))

    if user_input and user_input.strip():
        if not st.session_state.chatbot:
            st.error("Please initialize the chatbot first!")
        else:
            # Add user message to history immediately; mark pending assistant reply
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.pending_response = True
            # Rotate placeholder for the next prompt
            try:
                _next_placeholder()
            except Exception:
                pass
            st.rerun()
    col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
    
    with col2:
        
        # Responsive chat styles using viewport-relative sizing
        st.markdown("""
        <style>
        .chat-wrapper { width: min(100%, 100vw); max-width: min(900px, 92vw); margin: 0 auto; }
        .chat-scroll { max-height: 70vh; overflow-y: auto; padding: 10px; margin-bottom: 20px; }
        .stForm { position: sticky; bottom: 0; background: transparent !important; padding: 10px 0; border-top: 1px solid rgba(0,0,0,0.08); }

        /* Right-aligned user message row */
        .chat-row { display: flex; align-items: flex-end; gap: 8px; margin: 6px 0; }
        .chat-row.user { justify-content: flex-end; }
        .chat-row.assistant { justify-content: flex-start; }

        .chat-row .bubble {
          max-width: 72ch;
          padding: 10px 12px;
          border-radius: 14px;
          line-height: 1.35;
          word-wrap: break-word;
          word-break: break-word;
          box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        }
        .chat-row.user .bubble { background: #ffffffcc; border: 1px solid rgba(0,0,0,0.06); }
        .chat-row.assistant .bubble { background: #ffffffe6; border: 1px solid rgba(0,0,0,0.06); }

        .chat-row .avatar { width: 28px; height: 28px; border-radius: 6px; flex: 0 0 28px; }
        .chat-row .avatar.user { background: #ff9800; }

        /* Ensure assistant avatar uses the full image without cropping */
        [data-testid="stChatMessageAvatar"] { overflow: visible !important; }
        [data-testid="stChatMessageAvatar"] img {
          object-fit: contain !important;
          width: 100% !important;
          height: 100% !important;
          border-radius: 6px !important; /* keep slight rounding to match style */
          background: transparent !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Custom assistant avatar: embed local PNG as data URL so Streamlit renders it
        assistant_avatar = None
        try:
            icon2 = os.path.join(BASE_DIR, "assets", "icon2.png")
            icon1 = os.path.join(BASE_DIR, "assets", "icon.png")
            icon_path = icon2 if os.path.exists(icon2) else (icon1 if os.path.exists(icon1) else "")
            if icon_path:
                import base64
                with open(icon_path, "rb") as _f:
                    _b64 = base64.b64encode(_f.read()).decode("utf-8")
                assistant_avatar = f"data:image/png;base64,{_b64}"
        except Exception:
            assistant_avatar = None
        
        # If a user message was just submitted, add an assistant placeholder now
        if st.session_state.get('pending_response'):
            st.session_state.chat_history.append({"role": "assistant", "content": ""})
            st.session_state.pending_response = False

        # Display chat history in scrollable container
        st.markdown('<div class="chat-wrapper"><div class="chat-scroll">', unsafe_allow_html=True)
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "assistant" and message["content"] == "" and i == len(st.session_state.chat_history) - 1:
                    # This is an empty assistant response that needs streaming
                    with st.chat_message("assistant", avatar=assistant_avatar):
                        message_placeholder = st.empty()
                        
                        # Show loading spinner before streaming starts
                        line, emoji = random.choice(list(THINKING_PHRASES.items()))
                        _html = """
                                <div class="bambi-thinking-wrap">
                                  <div class="bambi-thinking">
                                    <span class="text">{LINE}</span>
                                    <span class="deer">{EMOJI}</span>
                                  </div>
                                </div>
                                <style>
                                  .bambi-thinking-wrap {
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    width: 100%;
                                    min-height: 56px; /* make space so vertical centering is visible */
                                    text-align: center;
                                  }
                                  @keyframes bambi-gallop {
                                    0% { transform: translateX(0) rotate(0deg); }
                                    25% { transform: translateX(6px) rotate(2deg); }
                                    50% { transform: translateX(12px) rotate(0deg); }
                                    75% { transform: translateX(6px) rotate(-2deg); }
                                    100% { transform: translateX(0) rotate(0deg); }
                                  }
                                  @keyframes bambi-pulse {
                                    0% { opacity: 0.6; }
                                    50% { opacity: 1; }
                                    100% { opacity: 0.6; }
                                  }
                                  .bambi-thinking {
                                    display: inline-flex;
                                    align-items: center;
                                    justify-content: center;
                                    gap: 8px;
                                    font-style: italic;
                                    color: #44A644;
                                    padding: 6px 8px;
                                    border-radius: 8px;
                                    background: transparent !important;
                                    border: none !important;
                                  }
                                  .bambi-thinking .deer {
                                    display: inline-block;
                                    font-size: 1.15rem;
                                    animation: bambi-gallop 1.2s ease-in-out infinite;
                                    filter: saturate(1.1);
                                  }
                                  .bambi-thinking .text {
                                    animation: bambi-pulse 1.6s ease-in-out infinite;
                                  }
                                </style>
                        """
                        message_placeholder.markdown(_html.replace("{LINE}", html_lib.escape(line)).replace("{EMOJI}", emoji), unsafe_allow_html=True)
                        
                        full_response = ""
                        
                        try:
                            # Get the user input from the previous message
                            user_input = st.session_state.chat_history[i-1]["content"]
                            
                            # Small delay to show the loading indicator
                            import time
                            time.sleep(0.5)
                            
                            # Stream the response word by word
                            for token in st.session_state.chatbot.chat_stream(user_input):
                                full_response += token
                                message_placeholder.markdown(full_response + "▌")
                            
                            # Remove cursor and finalize
                            message_placeholder.markdown(full_response)
                            
                            # Update the history with complete response
                            st.session_state.chat_history[i]["content"] = full_response
                            
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error: {str(e)}"
                            message_placeholder.markdown(error_msg)
                            st.session_state.chat_history[i]["content"] = error_msg
                else:
                    # Regular message display - use st.chat_message to keep consistent styling
                    if message["content"]:  # Only display non-empty messages
                        if message["role"] == "assistant":
                            with st.chat_message("assistant", avatar=assistant_avatar):
                                st.markdown(message["content"])
                        else:
                            # Custom right-aligned user bubble with avatar on the right
                            st.markdown(
                                f'''<div class="chat-row user">
                                      <div class="bubble">{message["content"]}</div>
                                      <div class="avatar user"></div>
                                    </div>''',
                                unsafe_allow_html=True,
                            )
        # Close responsive wrappers
        st.markdown('</div></div>', unsafe_allow_html=True)

        # Input handled above (outside columns)
        
        # # Clear chat button
        # if st.button("🗑️ Clear Chat History"):
        #     st.session_state.chat_history = []
        #     st.rerun()

if __name__ == "__main__":
    main()
