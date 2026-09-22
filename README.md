# 🤖 AI ASTRA

**Autonomous Interlinked Agentic Intelligence System**

A collaborative multi-agent AI assistant built with **Streamlit**, **LangGraph**, **LangChain**, and **Groq LLM**.

---

## 🏗️ Multi-Agent Architecture

```
                        USER
                         │
                         ▼
                  ORCHESTRATOR AGENT
                         │
              ┌──────────┼──────────┐
              ▼          │          ▼
          RESEARCH    (routing)   CODING
           AGENT                   AGENT
              │                     │
              └──── research ───────┘
                                    │
                                    ▼
                             TESTING AGENT
                                    │
                         ┌──────────┴──────────┐
                    PASS │                FAIL  │ (retry ≤ 3)
                         ▼                      ▼
                  FINAL RESPONSE        CODING AGENT (fix)
                                               │
                                               ▼
                                        TESTING AGENT
                                               │
                                         (repeat...)
```

### Agents

| Agent | Role |
|---|---|
| 🧠 **Orchestrator** | Understands the request, decides routing, composes the final response |
| 🔬 **Research Agent** | Analyses requirements, recommends algorithms, provides technical guidance to the Coding Agent |
| 💻 **Coding Agent** | Generates or fixes code using research findings and testing feedback |
| 🧪 **Testing Agent** | Reviews code, generates test cases, detects bugs, returns structured feedback |
| 📊 **Strategy Agent** | Handles business strategy, SWOT analysis, startup planning |
| 📎 **File Analysis** | Analyses uploaded PDFs and source-code files |

### Shared State

Every agent reads and writes a shared `AgentState` dictionary:

```python
{
  "user_request":         str,   # original user prompt
  "programming_language": str,   # user-selected language
  "needs_research":       bool,  # orchestrator decision
  "research_result":      str,   # Research Agent output
  "generated_code":       str,   # Coding Agent output
  "test_results":         str,   # Testing Agent report
  "bugs_found":           bool,  # testing verdict
  "retry_count":          int,   # fix iterations (max 3)
  "final_response":       str,   # orchestrator summary
  "activity_log":         list,  # per-step status messages
}
```

### Coding Pipeline (LangGraph StateGraph)

```
orchestrator ──► [research?] ──► coding ──► testing
                                    ▲           │
                                    │  bugs +   │ pass or
                                    └── retry ◄─┘ max retries
                                                │
                                         final_response ──► END
```

---

## Features

| Capability | Description |
|---|---|
| 🧠 Orchestrator Agent | Routing, coordination, final response |
| 🔬 Research Agent | Technical analysis, algorithm recommendations |
| 💻 Coding Agent | Code generation with research context |
| 🧪 Testing Agent | Code review, test cases, bug detection, fix loop |
| 📊 Strategy Agent | SWOT analysis, startup strategy, business planning |
| 🧠 Memory | Persistent conversation memory via LangGraph MemorySaver |
| 📎 File Analysis | Upload PDFs and code/text files for AI-powered analysis |
| 💬 Welcome Screen | Dynamic greeting with suggestion cards |
| 📡 Agent Activity | Live sidebar panel showing per-step agent status |

---

## File Upload & Analysis

Drop any supported file into the **Upload File** panel in the sidebar, then type a request in the chat:

- `summarize this document`
- `explain the code`
- `review and find bugs`
- `what does this function do?`

**Supported formats:** PDF, TXT, MD, CSV, JSON, YAML, Python, JS, TS, Java, C, C++, C#, Go, Rust, Ruby, PHP, Swift, Kotlin, R, Shell, SQL, HTML, CSS, and more.

---

## Agent Activity Panel

The sidebar shows real-time status during coding tasks:

```
📡 Agent Activity
─────────────────
🧠 Orchestrator: Analysing request…
🔬 Research Agent: Investigating technical approach…
🔬 Research Agent: Research complete.
💻 Coding Agent: Generating code…
💻 Coding Agent: Code generated. Passing to Testing Agent.
🧪 Testing Agent: Reviewing code…
🧪 Testing Agent: All tests passed ✅
🧠 Orchestrator: Composing final response…
✅ Task completed.
```

---

## Quick Start

### 1. Clone & enter the project

```bash
git clone <repo-url>
cd ibm-hackathon-project
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 5. Run the app

```bash
streamlit run ai_astra_ui.py
```

The app opens at **http://localhost:8501**.

---

## Project Structure

```
ibm-hackathon-project/
├── ai_astra_ui.py         # Main Streamlit application
├── agents/
│   ├── __init__.py
│   └── multi_agent.py     # Collaborative multi-agent pipeline
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not committed)
└── README.md
```

---

## Tech Stack

- **UI:** Streamlit
- **LLM:** Groq (`openai/gpt-oss-120b`)
- **Agent Framework:** LangGraph (StateGraph) + LangChain
- **Memory:** LangGraph MemorySaver
- **PDF Extraction:** PyMuPDF with pypdf fallback
- **Environment:** python-dotenv

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key |

---

## License

MIT
