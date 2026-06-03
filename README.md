# 🤖 Hey Neo — Your Local Linux System Assistant

> **"What's installed? What's running? What does this error mean?"**  
> Neo knows your machine better than you do.

---

## 🎯 Goal & Vision

**Hey Neo** is a fully local, privacy-first AI assistant built specifically for Linux power users. Its core purpose is to let you query your own system — packages, hardware, services, environments, and configurations — using plain natural language, with **zero data leaving your machine**.

The long-term vision is an **agentic RAG system** that:

- **Understands your entire system context** — not just Google's knowledge of Linux in general, but *your* specific machine: what's installed, what's running, what version, where it is.
- **Reasons over that context** like a senior sysadmin who has memorized your setup.
- **Falls back to the web** only when necessary, clearly telling you the source.
- **Evolves with your machine** — re-ingest at any time to keep the knowledge base up to date.

This is not a generic chatbot. Neo is purpose-built to answer questions like:

- *"Is Node.js installed and which version?"*
- *"What GPU do I have and is the driver loaded?"*
- *"Which Python virtual envs do I have under conda?"*
- *"What services are currently running on port 8080?"*
- *"What's the latest stable kernel version?"* ← web fallback

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│   Query Rewriter            │  qwen2.5:1.5b (fast, lightweight)
│   rewrite_query()           │  Normalizes, denoises, clarifies intent
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Agentic Loop              │  qwen3:14b (reasoning model)
│   answering_agent()         │  ReAct pattern — decides which tools to use
└──────┬──────────────────────┘
       │
       ├──► bm25_search_tool      →  Qdrant BM25_search     (sparse/keyword)
       ├──► similarity_search_tool→  Qdrant Similarity_search (dense/semantic)
       ├──► package_search_tool   →  Qdrant packages          (dense/semantic)
       └──► web_search_tool       →  Tavily API               (web fallback)
```

### Retrieval Strategy

| Tool | Index | Method | Best For |
|---|---|---|---|
| `bm25_search_tool` | `BM25_search` | Sparse (BM25, `Qdrant/bm25`) | Exact package/path/library names |
| `similarity_search_tool` | `Similarity_search` | Dense cosine (`mxbai-embed-large`) | Hardware, services, system state |
| `package_search_tool` | `packages` | Dense cosine (`mxbai-embed-large`) | APT package discovery |
| `web_search_tool` | Tavily API | Web search | Latest/external information |

The agent follows a strict **local-first, web-last** priority order. Web search is only triggered when all local tools fail or the query explicitly needs external data.

---

## 📁 Project Structure

```
hey-neo/
│
├── main.py                  # Entry point — takes user query, runs pipeline
├── agents.py                # Agentic loop — LLM + tools wired via LangChain
├── tools.py                 # LangChain @tool definitions + query rewriter
├── retrival.py              # Low-level Qdrant retrieval functions
│
├── ingest/                  # One-time ingestion scripts (run before first use)
│   ├── bm25_ingest.py       # Chunks & ingests package/path data → BM25_search
│   ├── similarity_ingest.py # Chunks & ingests hardware/system data → Similarity_search + packages
│   └── ingest_chat.py       # (Placeholder) Future chat history ingestion
│
├── docs/                    # Raw system snapshot files (your machine's data)
│   ├── apt_packages.txt     # dpkg -l output
│   ├── apt_manual.txt       # Manually installed packages
│   ├── python_packages.txt  # pip list output
│   ├── flatpak_packages.txt # flatpak list output
│   ├── conda_info.txt       # conda info output
│   ├── shell_config.txt     # ~/.bashrc or ~/.zshrc
│   ├── hardware_info.txt    # inxi / lshw output
│   ├── system_info.txt      # uname, lsb_release, etc.
│   ├── services.txt         # systemctl list-units output
│   ├── network_info.txt     # ip a, ss output
│   ├── docker_info.txt      # docker ps, images output
│   └── ollama_info.txt      # ollama list output
│
├── .env                     # API keys (see setup)
├── .gitignore
├── pyrightconfig.json
├── requirements.txt
└── agent.md                 # Learning tracker / dev log
```

---

## ⚙️ Prerequisites

### 1. Ollama (Local LLM + Embeddings)

Install [Ollama](https://ollama.com) and pull the required models:

```bash
# Main reasoning agent
ollama pull qwen3:14b

# Query rewriter (fast, lightweight)
ollama pull qwen2.5:1.5b

# Embedding model (1024-dim, used for semantic search)
ollama pull mxbai-embed-large:latest
```

### 2. Qdrant (Vector Database)

Run Qdrant locally via Docker:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Qdrant will be available at `http://localhost:6333`.

### 3. Tavily API Key (Web Search Fallback)

Get a free key at [tavily.com](https://tavily.com) and add it to your `.env` file:

```env
TAVILY_API_KEY=tvly-your-key-here
```

---

## 🚀 Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/kartik-dayatar/hey-neo.git
cd hey-neo

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your TAVILY_API_KEY

# 5. Populate docs/ with your system snapshots (see below)

# 6. Run ingestion (one-time setup)
python -m ingest.bm25_ingest
python -m ingest.similarity_ingest
```

---

## 📸 Populating `docs/` — Your System Snapshots

Neo's knowledge comes from snapshots of your machine. Run these commands and save the output to the corresponding files in `docs/`:

```bash
# APT packages
dpkg -l > docs/apt_packages.txt

# Manually installed packages
apt-mark showmanual > docs/apt_manual.txt

# Python packages (active env)
pip list > docs/python_packages.txt

# Flatpak packages
flatpak list > docs/flatpak_packages.txt

# Conda environments & system info
conda info && conda env list > docs/conda_info.txt

# Shell configuration (aliases etc.)
cat ~/.bashrc > docs/shell_config.txt     # or ~/.zshrc

# Hardware info (requires inxi)
inxi -Fxz > docs/hardware_info.txt

# System info
uname -a && lsb_release -a > docs/system_info.txt

# Running services
systemctl list-units --type=service --state=running > docs/services.txt

# Network info
ip a && ss -tulnp > docs/network_info.txt

# Docker state
docker ps && docker images > docs/docker_info.txt

# Ollama models
ollama list > docs/ollama_info.txt
```

---

## 💬 Usage

```bash
python main.py
```

```
What the problem dude neo is here......!
> is ffmpeg installed and what version?

====================================================================================================
From your installed packages: ffmpeg is installed, version 7:6.1.1-3ubuntu5.
```

Neo will:
1. **Rewrite** your query for precision
2. **Decide** which retrieval tools to use (BM25 → semantic → web, in that order)
3. **Retrieve** relevant context from Qdrant
4. **Answer** concisely, citing the source

---

## 🔍 How the Ingestion Works

### BM25 Ingestion (`ingest/bm25_ingest.py`)

Builds a **sparse keyword index** optimized for exact lookups:

- Parses APT, manual, Python, Flatpak, Conda, and shell config files
- Each package/item becomes one chunk with structured payload
- Embeds using `fastembed` with the `Qdrant/bm25` sparse model
- Upserts 4800+ points into the `BM25_search` Qdrant collection

### Semantic Ingestion (`ingest/similarity_ingest.py`)

Builds a **dense vector index** for meaning-based queries:

- Parses 6 hardware/system files using a `=== SECTION ===` header parser
- Splits content into overlapping 150-word chunks (35-word overlap) with noise filtering
- Embeds using Ollama's `mxbai-embed-large` (1024 dimensions)
- Maintains two collections: `Similarity_search` (system info) and `packages` (APT packages)

---

## 🗺️ Roadmap

- [x] BM25 sparse ingestion pipeline
- [x] Semantic dense ingestion pipeline  
- [x] Multi-retrieval query engine
- [x] Query rewriting with lightweight LLM
- [x] LangChain `@tool` definitions
- [x] Agentic loop (ReAct pattern with `qwen3:14b`)
- [x] Web search fallback via Tavily
- [ ] Chat history ingestion (`ingest_chat.py`)
- [ ] Automated snapshot refresh script
- [ ] TUI / interactive shell interface
- [ ] Multi-machine support

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| LLM (Agent) | `qwen3:14b` via Ollama + LangChain |
| LLM (Rewriter) | `qwen2.5:1.5b` via Ollama |
| Embeddings | `mxbai-embed-large:latest` via Ollama |
| Sparse Embeddings | `Qdrant/bm25` via fastembed |
| Vector DB | Qdrant (local, Docker) |
| Web Search | Tavily API |
| Agent Framework | LangChain (`langchain`, `langchain_ollama`) |
| Language | Python 3.12+ |

---

## 🔐 Privacy

All retrieval and reasoning is **100% local** by default. The only external call made is `web_search_tool` via Tavily — and only when the agent explicitly decides local tools cannot answer the query. No system data is ever sent externally.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
