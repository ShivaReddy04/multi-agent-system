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
- LangChain
- OpenAI GPT
- Pydantic
- JSON Output Parser

### Search & Retrieval
- Tavily Search API
- Web Scraping

### Frontend
- Streamlit

### Development
- Git
- GitHub
- VS Code

---

## 📂 Project Structure

```
multi-agent-research-system/
│
├── agents/
│   ├── search_agent.py
│   ├── reader_agent.py
│   └── coordinator.py
│
├── chains/
│
├── prompts/
│
├── models/
│
├── tools/
│
├── utils/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
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
- PDF report export
- Research history
- Vector database integration
- RAG-based knowledge retrieval
- Multi-model LLM support
- Authentication & user management

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
python -m venv venv
```

### Activate Environment

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key
TAVILY_API_KEY=your_api_key
```

### Run Streamlit

```bash
streamlit run streamlit_app.py
```

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
