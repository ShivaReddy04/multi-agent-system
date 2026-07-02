# 🔍 Multi-Agent Research System

An AI-powered research assistant that leverages multiple specialized agents to automate the research workflow. The system searches the web, extracts relevant information, analyzes content, and generates structured, citation-backed reports using Large Language Models (LLMs).

---

## 🚀 Features

- 🤖 Multi-Agent Architecture
- 🌐 Real-time Web Search
- 📄 Intelligent Content Extraction
- 🧠 LLM-powered Research Analysis
- 📑 Structured Report Generation
- ✅ JSON & Pydantic-based Output Validation
- 📊 Streamlit Interactive UI
- 🔄 Modular Agent Workflow
- ⚡ FastAPI Backend Support
- 📚 Source Attribution & Citations
- 📄 PDF Report Export
- 🕘 Research History
- 🗂️ Vector Database Integration (Chroma)
- 🔎 RAG-based Knowledge Retrieval (chat with reports)
- 🔀 Multi-model LLM Support (GitHub Models + Groq fallback)

---

## 🏗️ Architecture

```
                User Query
                     │
                     ▼
            Research Coordinator
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  Search Agent             Reader Agent
        │                         │
        ▼                         ▼
   Web Search API          Content Extraction
        │                         │
        └────────────┬────────────┘
                     ▼
              Report Generator
                     │
                     ▼
             Structured Response
```

---

## 🛠️ Tech Stack

### Backend
- Python
- FastAPI

### AI & LLM
- LangChain / LangGraph
- GitHub Models (OpenAI gpt-4o-mini) — primary
- Groq (Llama 3.3 70B) — automatic fallback
- Pydantic
- JSON Output Parser

### Search & Retrieval
- Tavily Search API
- Web Scraping
- Chroma vector store (RAG chat over reports)

### Frontend
- Streamlit

### Development
- Git
- GitHub
- VS Code

---

## 📂 Project Structure

```
multi-agent-system/
│
├── researchmind/
│   ├── agents/
│   │   ├── llm.py            # shared LLM config (GitHub Models + Groq fallback)
│   │   ├── search_agent.py   # web search agent (Tavily)
│   │   ├── reader_agent.py   # scrapes & extracts page content
│   │   ├── writer_agent.py   # drafts the structured report
│   │   ├── critic_agent.py   # reviews & scores the report
│   │   ├── chat_agent.py     # Q&A over a report
│   │   └── custom_parser.py
│   │
│   ├── app.py               # Streamlit UI
│   ├── api.py               # FastAPI backend
│   ├── pipeline.py          # shared research pipeline (used by UI + API)
│   ├── database.py          # SQLite research history
│   ├── vector_store.py      # Chroma vector store for RAG chat
│   ├── pdf_utils.py         # PDF report export
│   └── tools.py             # Tavily search tool
│
├── data/                    # SQLite DB + Chroma store (gitignored)
├── outputs/                 # generated PDF reports
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Workflow

1. User submits a research query.
2. Search Agent retrieves relevant web sources.
3. Reader Agent extracts and analyzes content.
4. LLM synthesizes findings.
5. Pydantic validates the structured output.
6. Streamlit displays the final report with citations.

---

## 📸 Screenshots

### Home Page

> Add screenshot here

### Research Results

> Add screenshot here

### Generated Report

> Add screenshot here

---

## 💡 Key Highlights

- Modular multi-agent architecture
- Structured AI responses using Pydantic
- Reliable JSON output parsing
- Extensible workflow for new agents
- Easy-to-maintain codebase
- Real-time internet research
- Source-backed responses

---

## 📈 Future Improvements

- Memory-enabled agents
- Multi-agent collaboration
- Authentication & user management
- Background jobs for long-running research requests

---

## 🖥️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/multi-agent-research-system.git
```

### Navigate

```bash
cd multi-agent-research-system
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

Create a `.env` file:

```env
# Primary LLM — GitHub Models (OpenAI gpt-4o-mini via an OpenAI-compatible endpoint)
GITHUB_TOKEN=your_github_token
# GITHUB_MODEL=openai/gpt-4o-mini   # optional override

# Fallback LLM — Groq free tier (used automatically when the primary is rate-limited)
GROQ_API_KEY=your_groq_api_key
# GROQ_MODEL=llama-3.3-70b-versatile   # optional override

# Web search
TAVILY_API_KEY=your_tavily_api_key
```

### Run Streamlit

```bash
streamlit run researchmind/app.py
```

---

## ⚡ Running the API

Alongside the Streamlit UI, the project exposes a **FastAPI** backend that wraps
the same multi-agent pipeline, so any client (a frontend, a script, another
service) can drive it over plain HTTP.

### Start the server

```bash
uvicorn researchmind.api:app --reload
```

The API runs at **http://localhost:8000**. Interactive Swagger docs — where you
can try every endpoint from the browser — are at **http://localhost:8000/docs**.

> ℹ️ A research run is slow (multiple LLM + search calls and a write/critique
> retry loop), so `POST /research` blocks until the pipeline finishes
> (roughly 30s–2min per request).

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness check |
| `POST` | `/research` | Run the full research pipeline for a topic |
| `GET`  | `/history?limit=20` | List past reports, newest first |
| `GET`  | `/reports/{id}` | Fetch a single stored report |
| `GET`  | `/reports/{id}/pdf` | Download a report as a PDF |

### Example request

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "quantum computing breakthroughs in 2025"}'
```

Example response:

```json
{
  "topic": "quantum computing breakthroughs in 2025",
  "report": "# Research Report ...",
  "feedback": "Score: 8/10 ...",
  "score": 8,
  "attempts": 1
}
```

> The Streamlit UI and the API can run at the same time on different ports.

---

## 🎯 Use Cases

- Academic Research
- Market Research
- Competitive Analysis
- Technology Research
- Literature Review
- Company Intelligence
- AI-assisted Knowledge Discovery

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository, open issues, and submit pull requests.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Shiva Panyala**

- GenAI Engineer
- Python Developer
- FastAPI | LangChain | LLMs | Multi-Agent Systems

---

⭐ If you found this project useful, please consider giving it a star!
