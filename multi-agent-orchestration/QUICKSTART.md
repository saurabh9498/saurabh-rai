# 🚀 Quick Start Guide

Get the Multi-Agent AI System running in minutes.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Required |
| OpenAI API Key | Or Anthropic API key |
| Docker (optional) | Easiest setup method |
| 4GB RAM | For embeddings model |

---

## Option 1: Docker (Recommended)

The fastest way to get everything running.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/multi-agent-ai-system.git
cd multi-agent-ai-system

# 2. Set up environment variables
cp .env.example .env

# 3. Edit .env and add your API key
#    Open .env in your editor and set:
#    OPENAI_API_KEY=sk-your-openai-key-here

# 4. Build and run with Docker Compose
cd docker
docker-compose up --build

# 5. Access the applications:
#    - Streamlit UI:  http://localhost:8501
#    - FastAPI:       http://localhost:8000
#    - API Docs:      http://localhost:8000/docs
```

### Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI backend |
| `ui` | 8501 | Streamlit interface |
| `chromadb` | 8001 | Vector database (internal) |

---

## Option 2: Local Development

For development and debugging.

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/multi-agent-ai-system.git
cd multi-agent-ai-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate        # macOS/Linux
# OR
venv\Scripts\activate           # Windows
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (required)
# At minimum, set your OpenAI API key:
#   OPENAI_API_KEY=sk-your-openai-key-here
```

### Step 4: Generate Sample Data (Recommended)

```bash
python scripts/generate_sample_data.py
```

This creates:
- Knowledge base documents (policies, technical docs, FAQs)
- Sample test queries
- Product catalog data

### Step 5: Run the API Server

```bash
uvicorn src.api.main:app --reload --port 8000
```

### Step 6: Run the UI (New Terminal)

```bash
# Open a new terminal, activate venv, then:
streamlit run ui/streamlit_app.py
```

### Access Points

- **Streamlit UI**: http://localhost:8501
- **FastAPI**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Option 3: Python Script Demo

Minimal code to test the system programmatically.

```python
# demo.py
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"

from src.agents.orchestrator import Orchestrator
from src.rag.pipeline import RAGPipeline

# Initialize RAG pipeline and ingest documents
rag = RAGPipeline()
rag.ingest_directory("data/knowledge_base/")

# Initialize the orchestrator
orchestrator = Orchestrator()

# Execute a query
result = orchestrator.execute(
    query="What is the remote work policy?",
    context=rag.retrieve("remote work policy")
)

print(result.response)
print(f"Agent used: {result.agent}")
print(f"Sources: {result.sources}")
```

Run with:
```bash
python demo.py
```

---

## Option 4: Jupyter Notebook

Interactive exploration of the system.

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/demo.ipynb
```

The demo notebook covers:
1. ✅ System initialization
2. ✅ Document ingestion
3. ✅ RAG retrieval
4. ✅ Single agent execution
5. ✅ Multi-agent workflows
6. ✅ Tool usage examples

---

## Verify Installation

### Check API Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "agents": ["orchestrator", "research", "analyst", "code"]
}
```

### Test a Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the remote work policy?"}'
```

### List Available Agents

```bash
curl http://localhost:8000/api/v1/agents
```

---

## Project Structure

```
multi-agent-ai-system/
│
├── src/                      # Source code
│   ├── agents/               # 🤖 AI Agents
│   │   ├── orchestrator.py   #    Main coordinator
│   │   ├── research_agent.py #    Information retrieval
│   │   ├── analyst_agent.py  #    Data analysis
│   │   └── code_agent.py     #    Code generation
│   │
│   ├── rag/                  # 📚 RAG Pipeline
│   │   ├── pipeline.py       #    Main RAG orchestration
│   │   ├── retriever.py      #    Document retrieval
│   │   ├── embeddings.py     #    Embedding generation
│   │   └── chunker.py        #    Document chunking
│   │
│   ├── core/                 # ⚙️ Core utilities
│   │   ├── config.py         #    Settings management
│   │   ├── llm.py            #    LLM abstraction
│   │   └── context.py        #    Context management
│   │
│   ├── tools/                # 🔧 Agent tools
│   │   ├── web_search.py     #    Web search tool
│   │   ├── database.py       #    Database queries
│   │   └── code_executor.py  #    Code execution
│   │
│   └── api/                  # 🌐 REST API
│       ├── main.py           #    FastAPI app
│       ├── routes.py         #    API endpoints
│       └── schemas.py        #    Pydantic models
│
├── ui/                       # 💻 Streamlit UI
│   └── streamlit_app.py
│
├── data/                     # 📁 Data files
│   ├── knowledge_base/       #    Documents for RAG
│   └── sample/               #    Test data
│
├── config/                   # ⚡ Configuration
│   └── agents.yaml           #    Agent settings
│
├── docker/                   # 🐳 Docker setup
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── tests/                    # ✅ Test suites
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── notebooks/                # 📓 Jupyter notebooks
│   └── demo.ipynb
│
└── scripts/                  # 🔨 Utility scripts
    └── generate_sample_data.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/query` | POST | Execute a query |
| `/api/v1/query/stream` | POST | Stream response |
| `/api/v1/agents` | GET | List agents |
| `/api/v1/agents/{id}` | GET | Get agent details |
| `/api/v1/ingest` | POST | Ingest documents |
| `/api/v1/search` | POST | Search knowledge base |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc documentation |

---

## Environment Variables

Key variables in `.env`:

```bash
# LLM Configuration
LLM_PROVIDER=openai              # openai, anthropic, or local
LLM_MODEL=gpt-4-turbo            # Model to use
OPENAI_API_KEY=sk-xxx            # Your OpenAI key

# Vector Store
VECTOR_STORE=chromadb            # chromadb, pinecone, or faiss
CHROMA_PERSIST_DIR=./data/chroma

# RAG Settings
CHUNK_SIZE=512                   # Document chunk size
TOP_K_RETRIEVAL=10               # Documents to retrieve
```

See `.env.example` for all available options.

---

## Troubleshooting

### "Module not found" errors

```bash
# Make sure you're in the project root and venv is activated
pip install -e .
```

### API key errors

```bash
# Verify your key is set
echo $OPENAI_API_KEY

# Or check .env file
cat .env | grep OPENAI
```

### Port already in use

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn src.api.main:app --port 8001
```

### Docker issues

```bash
# Clean rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

---

## Next Steps

1. **Explore the UI**: Try different queries in the Streamlit interface
2. **Add your documents**: Place files in `data/knowledge_base/`
3. **Run tests**: `pytest tests/`
4. **Read the docs**: Check `docs/` for architecture details
5. **Customize agents**: Edit `config/agents.yaml`

---

## Getting Help

- 📖 [Architecture Guide](docs/architecture.md)
- 🔌 [API Reference](docs/api_reference.md)
- 🚀 [Deployment Guide](docs/deployment.md)
- 🐛 [Open an Issue](https://github.com/your-username/multi-agent-ai-system/issues)

---

Happy building! 🎉
