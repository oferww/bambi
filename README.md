# 🦌 Bambi - Personal AI Assistant

A personalized chatbot that knows about you through RAG (Retrieval-Augmented Generation) using your photos, metadata, and memories. Built with Streamlit, LangChain, Cohere, and ChromaDB.

## ✨ Features

- **Photo Analysis**: Automatically extracts metadata from photos (date, location, camera info)
- **Memory Storage**: Add personal memories and experiences as text
- **RAG System**: Uses Cohere embeddings and ChromaDB for intelligent retrieval
- **Beautiful UI**: Modern Streamlit interface with chat functionality
- **Docker Support**: Fully containerized for easy deployment
- **Conversational AI**: Powered by Cohere's Command model

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Cohere API key (get one at [cohere.ai](https://cohere.ai/))

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd bambi
   ```

2. **Set up environment variables**
   ```bash
   cp env.example .env
   ```
   Edit `.env` and add your Cohere API key:
   ```
   COHERE_API_KEY=your_cohere_api_key_here
   ```

3. **Add your data**
   ```bash
   mkdir -p data/uploads/photos
   # Copy your photos to data/uploads/photos/
   # Optional: place PDFs under data/uploads/pdfs, CSVs under data/uploads/csvs, JSON under data/uploads/json
   ```

4. **Run locally**
   ```bash
   pip install -r requirements.txt
   streamlit run frontend/app.py
   ```

5. **Access the application**
   Open your browser and go to `http://localhost:8501`

## 📁 Project Structure

```
bambi/
├── backend/
│   ├── chatbot.py              # Chatbot (uses RAGSystem)
│   ├── rag_system.py           # RAG system (Cohere + ChromaDB)
│   ├── ingestion/
│   │   └── dispatcher.py       # Central ingestion entrypoints for UI
│   └── utils/                  # PDF/photo/JSON utilities
├── frontend/
│   └── app.py                  # Streamlit UI
├── dev_tools/                  # Dev-only CLI scripts (excluded from Docker)
├── tests/
│   └── backend/
│       ├── unit/
│       └── integration/
├── data/
│   ├── uploads/
│   │   ├── photos/
│   │   ├── pdfs/
│   │   ├── csvs/
│   │   └── json/
│   ├── embeddings/             # ChromaDB storage
│   └── embeddings_dump.csv     # CSV dump of collection
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker build
├── .dockerignore               # Excludes dev_tools/ and tests/
├── pytest.ini                  # Pytest config
├── .env                        # Environment variables (gitignored)
└── README.md                   # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
# Required
COHERE_API_KEY=your_cohere_api_key_here

# Optional UI/RAG settings
OFERGPT_HIDE_SIDEBAR=0
OFERGPT_RAG_CHUNK_SIZE=1500
OFERGPT_RAG_CHUNK_OVERLAP=250

# Optional: run one-time full S3 sync on server startup
OFERGPT_S3_SYNC_ON_START=0
S3_BUCKET=
S3_PREFIX=
# Preferred endpoint variable (e.g., Cloudflare R2)
S3_ENDPOINT=
# Legacy (still supported as fallback)
S3_ENDPOINT_URL=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=
```

### Supported Photo Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- TIFF (.tiff)

## 📸 Photo Metadata Extraction

The system automatically extracts:

- **Date taken** (from EXIF data)
- **GPS coordinates** (location)
- **Camera information** (model, make, settings)
- **Image properties** (dimensions, format)

## 💬 Usage

### 1. Initialize the Chatbot
- Click "🔄 Initialize Chatbot" in the sidebar
- Ensure your Cohere API key is configured

### 2. Add Files to Knowledge Base
- Place your files under `data/uploads/` subfolders:
  - Photos → `data/uploads/photos/`
  - PDFs → `data/uploads/pdfs/`
  - CSVs → `data/uploads/csvs/`
  - JSON/NDJSON → `data/uploads/json/`
- Use the sidebar buttons to ingest/embed.

### 3. Add Personal Memories
- Use the "Add Memories" section in the sidebar
- Enter text memories about your experiences
- These will be added to the knowledge base

### 4. Start Chatting
- Ask questions about your photos and memories
- The AI will use RAG to find relevant information
- Get personalized responses about your life and experiences

## 🐳 Docker Deployment

### Local Development
```bash
# Build image
docker build -t bambi .

# Run with environment variables and local data volume
# Powershell (Windows):
docker run --rm -p 8501:8501 \
  --env-file .env \
  -v ${PWD}/data:/app/data \
  bambi

# Bash (macOS/Linux):
docker run --rm -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  bambi
```

## 🔍 Programmatic Notes

The primary interface is the Streamlit UI (`frontend/app.py`), which calls backend ingestion via `backend/ingestion/dispatcher.py` and persists to ChromaDB via `RAGSystem`.
Advanced users can import `RAGSystem` to add text memories or perform collection operations directly.

## 🛠️ Development

### Local Development Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**
   ```bash
   export COHERE_API_KEY=your_key
   ```

3. **Run locally**
   ```bash
   streamlit run app.py
   ```

### Adding New Features

- **New photo processors**: Extend `utils/photo_processor.py`
- **Additional RAG sources**: Modify `rag_system.py`
- **UI improvements**: Update `app.py` and CSS
- **New LLM models**: Update `chatbot.py`

## 🔒 Security Notes

- Never commit your `.env` file
- Keep your Cohere API key secure
- Photos are processed locally, not uploaded to external services
- ChromaDB data is stored locally in `data/embeddings/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Cohere](https://cohere.ai/) for the LLM and embeddings
- [Streamlit](https://streamlit.io/) for the web framework
- [LangChain](https://langchain.com/) for the RAG framework
- [ChromaDB](https://www.trychroma.com/) for vector storage

## 🆘 Troubleshooting

### Common Issues

1. **"Cohere API Key not found"**
   - Ensure your `.env` file exists and contains the API key
   - Check that the key is valid at [cohere.ai](https://cohere.ai/)

2. **"No photos found"**
   - Make sure photos are in `data/photos/`
   - Check file extensions are supported

3. **"Error processing photos"**
   - Ensure photos are not corrupted
   - Check file permissions

4. **Docker build fails**
   - Ensure Docker is running
   - Check internet connection for package downloads

### Getting Help

- Check container logs: `docker logs <container_id>`
- Verify environment variables: check `.env` and the `--env-file` used for `docker run`
- Test API key: Use Cohere's playground

---

**Made with ❤️ for personal AI assistants**
