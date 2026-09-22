# ============================================================
# AI ASTRA
# Autonomous Interlinked Agentic Intelligence System
# Streamlit UI Version
# ============================================================

import io
import os
import re

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from agents.multi_agent import run_multi_agent_pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI ASTRA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM UI
# ============================================================

# ── Dynamic theme CSS is injected after session state is ready ──
# (see THEME CSS block below, after session state initialisation)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error(
        "GROQ_API_KEY is missing. Add it to your .env file."
    )
    st.stop()


# ============================================================
# INITIALIZE LLM
# ============================================================

@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-120b",
        temperature=0,
    )


llm = get_llm()


# ============================================================
# MEMORY
# ============================================================

@st.cache_resource
def get_memory():
    return MemorySaver()


memory = get_memory()


# ============================================================
# RESEARCH TOOL
# ============================================================

class ResearchInput(BaseModel):
    topic: str = Field(
        description="Research topic or educational query"
    )


def research_tool_func(topic: str):
    prompt = f"""
You are an expert technical educator.

Explain the following topic clearly:

{topic}

Include:
- Overview
- Important concepts
- Applications
- Advantages
- Future scope

Use simple and understandable language.
"""

    response = llm.invoke(prompt)
    return response.content


research_tool = StructuredTool.from_function(
    func=research_tool_func,
    name="research_tool",
    description=(
        "Use this tool for educational explanations, "
        "AI concepts, programming concepts, technical "
        "learning, research topics, and knowledge queries."
    ),
    args_schema=ResearchInput,
)


# ============================================================
# CODING TOOL
# ============================================================

class CodingInput(BaseModel):
    task: str = Field(
        description="Programming or coding task"
    )

    language: str = Field(
        description="Programming language selected by the user"
    )


def coding_tool_func(task: str, language: str):
    prompt = f"""
You are an expert software engineer.

Programming language selected by the user:
{language}

Programming task:
{task}

STRICT RULES:

1. Generate ONLY {language} code.
2. Do NOT use another programming language.
3. Do NOT mix programming languages.
4. Do NOT provide multiple language versions.
5. Keep the code simple and beginner-friendly.
6. Make the code complete and executable.
7. Check syntax carefully.
8. Fix syntax and logical errors before returning.
9. Include required imports.
10. Include input/output handling when required.
11. Return ONLY the final {language} code.
12. Do NOT provide an explanation.
13. Do NOT provide pseudocode.
14. Do NOT say "Here is the code".
15. Do NOT put explanation outside the code.

Return only the final source code.
"""

    response = llm.invoke(prompt)
    return response.content


coding_tool = StructuredTool.from_function(
    func=coding_tool_func,
    name="coding_tool",
    description=(
        "Use this tool for programming, coding, debugging, "
        "algorithms, APIs, software design, and development "
        "tasks. The programming language MUST already be "
        "selected by the user. Generate code ONLY in that "
        "selected language."
    ),
    args_schema=CodingInput,
)


# ============================================================
# STRATEGY TOOL
# ============================================================

class StrategyInput(BaseModel):
    business_problem: str = Field(
        description="Business or startup strategy problem"
    )


def strategy_tool_func(business_problem: str):
    prompt = f"""
You are an expert business strategist.

Analyze this business problem:

{business_problem}

Include:
- SWOT analysis
- Growth strategy
- Risks
- Recommendations

Give practical and understandable advice.
"""

    response = llm.invoke(prompt)
    return response.content


strategy_tool = StructuredTool.from_function(
    func=strategy_tool_func,
    name="strategy_tool",
    description=(
        "Use this tool for startup ideas, business strategy, "
        "optimization, market analysis, planning, and "
        "entrepreneurial guidance."
    ),
    args_schema=StrategyInput,
)


# ============================================================
# TOOL REGISTRY
# ============================================================

tools = [
    research_tool,
    coding_tool,
    strategy_tool,
]


# ============================================================
# REACT AGENT
# ============================================================

@st.cache_resource
def get_agent():
    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
    )


agent_executor = get_agent()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_coding_task" not in st.session_state:
    st.session_state.pending_coding_task = None

if "selected_language" not in st.session_state:
    st.session_state.selected_language = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "ai-astra-streamlit-thread"

if "uploaded_file_content" not in st.session_state:
    st.session_state.uploaded_file_content = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "suggested_prompt" not in st.session_state:
    st.session_state.suggested_prompt = None

if "agent_activity" not in st.session_state:
    st.session_state.agent_activity = []

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True


# ============================================================
# THEME CSS  (injected dynamically based on dark_mode flag)
# ============================================================

_DM = st.session_state.dark_mode

# ── Dark mode colour tokens (original design) ──
# ── Light mode colour tokens (clean professional light) ──
_bg          = "#0b1020" if _DM else "#f0f4f8"
_sidebar_bg  = "#111827" if _DM else "#e8edf5"
_card_bg     = "#111827" if _DM else "#ffffff"
_card_border = "#263244" if _DM else "#c9d5e3"
_lang_border = "#334155" if _DM else "#b0bfce"
_text_main   = "#f9fafb" if _DM else "#0f172a"
_text_muted  = "#9ca3af" if _DM else "#475569"
_text_small  = "#9ca3af" if _DM else "#64748b"
_act_bg      = "#0d1626" if _DM else "#eef2fb"
_act_border  = "#1e3a5f" if _DM else "#93b4d8"
_act_title   = "#60a5fa" if _DM else "#1d4ed8"
_act_item    = "#d1d5db" if _DM else "#1e293b"
_footer_col  = "#6b7280" if _DM else "#64748b"
_input_bg    = "#111827" if _DM else "#ffffff"
_input_text  = "#f9fafb" if _DM else "#0f172a"
_input_border= "#334155" if _DM else "#94a3b8"
# extra dark-mode surfaces that Streamlit renders internally
_stapp_bg2   = "#0b1020" if _DM else "#f0f4f8"
_block_bg    = "#111827" if _DM else "#ffffff"
_block_text  = "#f9fafb" if _DM else "#0f172a"

st.markdown(
    f"""
    <style>
        /* ══════════════════════════════════════════
           FULL-PAGE BACKGROUND — override every
           Streamlit layer so dark mode stays dark
           regardless of the browser/Streamlit theme
           ══════════════════════════════════════════ */
        html, body {{
            background-color: {_bg} !important;
            color: {_text_main} !important;
        }}
        .stApp,
        .stApp > div,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > section,
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        .main .block-container {{
            background-color: {_bg} !important;
            color: {_text_main} !important;
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div {{
            background-color: {_sidebar_bg} !important;
        }}
        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] li {{
            color: {_text_main} !important;
        }}

        /* ── All text in main area ── */
        .stApp p, .stApp span, .stApp label,
        .stApp li, .stApp h1, .stApp h2,
        .stApp h3, .stApp h4, .stApp h5,
        .stMarkdown, .stMarkdown p,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {{
            color: {_text_main} !important;
        }}

        /* ── Streamlit internal block/widget backgrounds ── */
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"],
        [data-testid="column"],
        .stChatFloatingInputContainer,
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div {{
            background-color: {_bg} !important;
        }}

        /* ── Chat messages ── */
        .stChatMessage {{
            border-radius: 14px;
            background-color: {_block_bg} !important;
        }}
        [data-testid="stChatMessageContent"],
        [data-testid="stChatMessageContent"] *,
        [data-testid="stChatMessageContent"] p {{
            color: {_block_text} !important;
            background-color: transparent !important;
        }}

        /* ── Chat input bar ── */
        [data-testid="stChatInput"],
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] textarea {{
            background-color: {_input_bg} !important;
            color: {_input_text} !important;
            border-color: {_input_border} !important;
        }}
        [data-testid="stChatInput"] textarea::placeholder {{
            color: {_text_muted} !important;
        }}

        /* ── Select/input widgets ── */
        [data-testid="stSelectbox"] div,
        [data-baseweb="select"] div,
        [data-baseweb="input"] input {{
            background-color: {_input_bg} !important;
            color: {_input_text} !important;
            border-color: {_input_border} !important;
        }}

        /* ── Cards ── */
        .status-card {{
            padding: 14px 18px;
            border-radius: 12px;
            background: {_card_bg};
            border: 1px solid {_card_border};
            margin-bottom: 15px;
        }}
        .language-card {{
            padding: 18px;
            border-radius: 14px;
            background: {_card_bg};
            border: 1px solid {_lang_border};
            margin-top: 10px;
            margin-bottom: 15px;
        }}
        .language-card h3, .language-card p {{
            color: {_text_main} !important;
        }}
        .small-text {{
            color: {_text_small};
            font-size: 13px;
        }}

        /* ── Code ── */
        code {{
            font-size: 14px !important;
        }}

        /* ── Welcome Screen ── */
        .welcome-wrap {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 60px 20px 40px 20px;
            text-align: center;
            animation: fadeInUp 0.5s ease both;
        }}
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(22px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .welcome-greeting {{
            font-size: 32px;
            font-weight: 700;
            color: {_text_main} !important;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }}
        .welcome-sub {{
            font-size: 17px;
            color: {_text_muted} !important;
            margin-bottom: 40px;
        }}
        .suggestion-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
            justify-content: center;
            max-width: 720px;
        }}

        /* ── Agent Activity Panel ── */
        .activity-panel {{
            background: {_act_bg} !important;
            border: 1px solid {_act_border};
            border-radius: 12px;
            padding: 14px 16px;
            margin: 18px 0 10px 0;
            font-size: 13px;
            line-height: 1.7;
        }}
        .activity-title {{
            font-size: 13px;
            font-weight: 600;
            color: {_act_title} !important;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            text-transform: uppercase;
        }}
        .activity-item {{
            color: {_act_item} !important;
            padding: 2px 0;
        }}

        /* ── Title & subtitle ── */
        .astra-title {{
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 0;
            color: {_text_main} !important;
        }}
        .astra-subtitle {{
            color: {_text_muted} !important;
            font-size: 16px;
            margin-top: 0;
        }}

        /* ── Footer ── */
        .astra-footer {{
            color: {_footer_col} !important;
        }}

        /* ── Divider ── */
        hr {{
            border-color: {_card_border} !important;
        }}

        /* ── Captions ── */
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        .stMarkdown small {{
            color: {_text_small} !important;
        }}

        /* ── Info / Success / Error boxes ── */
        [data-testid="stAlert"] {{
            background-color: {_card_bg} !important;
            border-color: {_card_border} !important;
        }}
        [data-testid="stAlert"] p {{
            color: {_text_main} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LANGUAGE OPTIONS
# ============================================================

LANGUAGES = [
    "Python",
    "Java",
    "C",
    "C++",
    "JavaScript",
    "TypeScript",
    "Go",
    "Rust",
    "C#",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_code_response(text: str) -> str:
    """
    Remove Markdown code fences if the LLM returns them.
    The UI then displays only the source code.
    """

    text = text.strip()

    match = re.match(
        r"^```(?:[a-zA-Z0-9_+#.-]+)?\s*(.*?)\s*```$",
        text,
        flags=re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return text


def is_coding_request(user_input: str) -> bool:
    """
    Classify whether a request is coding-related.
    """

    prompt = f"""
Classify this user request.

User request:
{user_input}

Return ONLY one word:

CODING

or

GENERAL

CODING means the user wants programming code, debugging,
an algorithm implementation, a program, source code, or
software development implementation.

GENERAL means explanations, research, business questions,
normal conversation, or other non-code requests.
"""

    result = llm.invoke(prompt).content.strip().upper()

    return result.startswith("CODING")


def run_general_agent(user_input: str) -> str:
    """
    Run AI ASTRA while keeping the conversation context small
    to avoid Groq token-limit errors.
    """

    # Keep only recent messages to control token usage
    recent_messages = st.session_state.messages[-4:]

    response = agent_executor.invoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": """
You are AI ASTRA.

Autonomous Interlinked Agentic Intelligence System.

Capabilities:
- Autonomous reasoning
- Intelligent tool usage
- Memory-aware conversation
- Technical research
- Coding assistance
- Business strategy

Rules:
- Decide which tool is appropriate.
- Use tools whenever required.
- Use multiple tools when necessary.
- Give professional, clear answers.
- Never expose chain-of-thought.
- Never expose hidden reasoning.

Keep responses concise and useful.
"""
                }
            ]
            + [
                {
                    "role": message["role"],
                    "content": message["content"]
                }
                for message in recent_messages
            ],
        },
        config={
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        },
    )

    return response["messages"][-1].content


def run_coding_agent(task: str, language: str) -> str:
    """
    Run the ReAct coding agent after the language has
    already been selected by the user.
    """

    response = agent_executor.invoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": f"""
You are AI ASTRA Coding Agent.

The user selected this programming language:

{language}

The user's coding task is:

{task}

STRICT RULES:

1. Use the coding_tool.
2. Pass the exact selected language: {language}.
3. Generate ONLY {language} code.
4. Do NOT generate code in another language.
5. Do NOT generate multiple language versions.
6. Keep the code simple and beginner-friendly.
7. Make the code complete and executable.
8. Check syntax carefully.
9. Fix errors before returning the code.
10. Return ONLY the source code.
11. Do NOT explain the code.
12. Do NOT provide pseudocode.
13. Do NOT say "Here is the code".
""",
                },
                {
                    "role": "user",
                    "content": task,
                },
            ]
        },
        config={
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        },
    )

    return clean_code_response(
        response["messages"][-1].content
    )


# ============================================================
# FILE EXTRACTION HELPERS
# ============================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF file using PyMuPDF (fitz)."""
    try:
        import pymupdf as fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        text = "\n".join(pages).strip()
        return text if text else "[PDF contains no extractable text]"
    except Exception as fitz_err:
        # Fallback to pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [p.extract_text() or "" for p in reader.pages]
            text = "\n".join(pages).strip()
            return text if text else "[PDF contains no extractable text]"
        except Exception as pdf_err:
            return f"[Could not extract PDF text: {fitz_err} | {pdf_err}]"


# Supported plain-text / code file extensions
TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift",
    ".kt", ".r", ".sh", ".bat", ".ps1", ".sql", ".html", ".css",
    ".toml", ".ini", ".cfg", ".env",
}

MAX_FILE_CHARS = 20_000  # truncate very large files to avoid token limits


def extract_text_from_file(uploaded_file) -> tuple[str, str]:
    """
    Extract text from an uploaded Streamlit file object.

    Returns (content: str, file_type: str)  where file_type is
    "pdf" | "text" | "unsupported".
    """
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if name.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
        return text[:MAX_FILE_CHARS], "pdf"

    # Detect text/code file by extension
    ext = os.path.splitext(name)[1]
    if ext in TEXT_EXTENSIONS:
        try:
            text = file_bytes.decode("utf-8", errors="replace")
            return text[:MAX_FILE_CHARS], "text"
        except Exception as e:
            return f"[Could not decode file: {e}]", "text"

    return "[Unsupported file type]", "unsupported"


# ============================================================
# FILE ANALYSIS AGENT
# ============================================================

def run_file_analysis_agent(
    file_content: str,
    file_name: str,
    user_request: str,
) -> str:
    """
    Analyse an uploaded file using the LLM directly (no tool
    routing needed — the content is already in the prompt).
    """

    truncation_note = (
        "\n\n[Note: File was truncated to the first 20 000 characters.]"
        if len(file_content) >= MAX_FILE_CHARS
        else ""
    )

    prompt = f"""You are AI ASTRA — a highly capable AI assistant.

The user has uploaded a file: **{file_name}**

--- FILE CONTENT START ---
{file_content}{truncation_note}
--- FILE CONTENT END ---

User's request: {user_request}

Please fulfil the user's request based on the file content above.
Be thorough, structured, and professional.
"""

    response = llm.invoke(prompt)
    return response.content


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🤖 AI ASTRA")

    st.markdown(
        """
        **Autonomous Interlinked Agentic Intelligence**

        Multi-agent team:
        - 🧠 Orchestrator Agent
        - 🔬 Research Agent
        - 💻 Coding Agent
        - 🧪 Testing Agent
        - 📊 Strategy Agent
        - 📎 File Analysis
        """
    )

    st.divider()

    st.markdown("### 🟢 System Status")

    st.success("ALL AGENTS ONLINE")

    st.caption("Model: openai/gpt-oss-120b")
    st.caption("Framework: LangGraph + LangChain")
    st.caption("Architecture: Collaborative Multi-Agent")

    st.divider()

    # --------------------------------------------------------
    # AGENT ACTIVITY PANEL
    # --------------------------------------------------------

    st.markdown("### 📡 Agent Activity")

    if st.session_state.agent_activity:
        items_html = "".join(
            f'<div class="activity-item">{line}</div>'
            for line in st.session_state.agent_activity
        )
        st.markdown(
            f'<div class="activity-panel">'
            f'<div class="activity-title">Live Status</div>'
            f'{items_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Activity will appear here during a coding task.")

    st.divider()

    # --------------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------------

    st.markdown("### 📎 Upload File")
    st.caption(
        "Supports PDF, Python, JS, Java, C/C++, "
        "JSON, CSV, TXT, MD, and more."
    )

    uploaded_file = st.file_uploader(
        "Drag & drop or browse",
        type=[
            "pdf",
            "txt", "md", "csv", "json", "xml",
            "yaml", "yml", "toml", "ini", "cfg",
            "py", "js", "ts", "jsx", "tsx",
            "java", "c", "cpp", "h", "hpp",
            "cs", "go", "rs", "rb", "php",
            "swift", "kt", "r", "sh", "bat",
            "ps1", "sql", "html", "css",
        ],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        with st.spinner("Reading file…"):
            content, ftype = extract_text_from_file(uploaded_file)

        if ftype == "unsupported":
            st.error(
                f"'{uploaded_file.name}' is not a supported "
                "file type."
            )
        else:
            st.session_state.uploaded_file_content = content
            st.session_state.uploaded_file_name = (
                uploaded_file.name
            )
            st.success(
                f"✅ **{uploaded_file.name}** loaded "
                f"({len(content):,} chars)"
            )
            st.caption(
                "Now type your request in the chat — "
                "e.g. *summarize*, *explain*, *review*, "
                "*debug*, *find bugs*."
            )

    # Clear uploaded file
    if st.session_state.uploaded_file_content is not None:
        if st.button(
            "🗑️ Clear Uploaded File",
            use_container_width=True,
        ):
            st.session_state.uploaded_file_content = None
            st.session_state.uploaded_file_name = None
            st.rerun()

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.pending_coding_task = None
        st.session_state.selected_language = None
        st.session_state.uploaded_file_content = None
        st.session_state.uploaded_file_name = None
        st.session_state.agent_activity = []
        st.session_state.thread_id = (
            f"ai-astra-streamlit-{id(st.session_state)}"
        )
        st.rerun()


# ============================================================
# HEADER  (with dark/light mode toggle)
# ============================================================

_hcol_title, _hcol_spacer, _hcol_toggle = st.columns([6, 2, 1])

with _hcol_title:
    st.markdown(
        '<div class="astra-title">🤖 AI ASTRA</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="astra-subtitle">'
        'Autonomous Interlinked Agentic Intelligence System'
        '</div>',
        unsafe_allow_html=True,
    )

with _hcol_toggle:
    _toggle_label = "☀️ Light" if st.session_state.dark_mode else "🌙 Dark"
    if st.button(
        _toggle_label,
        key="theme_toggle",
        help="Switch between dark and light mode",
        use_container_width=True,
    ):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.divider()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message.get("type") == "code":
            st.code(
                message["content"],
                language=message.get(
                    "language",
                    "text",
                ),
            )
        else:
            st.markdown(message["content"])


# ============================================================
# WELCOME SCREEN  (shown only when chat is empty)
# ============================================================

if (
    not st.session_state.messages
    and st.session_state.pending_coding_task is None
):
    import datetime

    _hour = datetime.datetime.now().hour
    if _hour < 12:
        _greeting = "Good morning!"
    elif _hour < 18:
        _greeting = "Good afternoon!"
    else:
        _greeting = "Good evening!"

    st.markdown(
        f"""
        <div class="welcome-wrap">
            <div class="welcome-greeting">{_greeting}</div>
            <div class="welcome-sub">Where should we begin?</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _suggestions = [
        ("💡", "Ask the Agent", "What can you help me with today?"),
        ("📄", "Analyze a File", "Upload a file in the sidebar, then ask me to analyze it."),
        ("🔍", "Research a Topic", "Research the latest advances in artificial intelligence."),
        ("💻", "Help Me Code", "Write a Python function to reverse a linked list."),
    ]

    _cols = st.columns(len(_suggestions))

    for _col, (_icon, _label, _prompt) in zip(_cols, _suggestions):
        with _col:
            if st.button(
                f"{_icon}  {_label}",
                key=f"suggestion_{_label}",
                use_container_width=True,
            ):
                st.session_state.suggested_prompt = _prompt
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# CODING LANGUAGE SELECTION UI
# ============================================================

if st.session_state.pending_coding_task is not None:

    st.markdown(
        """
        <div class="language-card">
            <h3>💻 Programming Language</h3>
            <p class="small-text">
                Your request is coding-related.
                Select the language before AI ASTRA generates code.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_language = st.selectbox(
        "Select programming language",
        LANGUAGES,
        index=0,
        key="language_selector",
    )

    col1, col2 = st.columns([1, 5])

    with col1:

        generate_clicked = st.button(
            "Generate Code",
            type="primary",
            use_container_width=True,
        )

    if generate_clicked:

        task = st.session_state.pending_coding_task

        st.session_state.selected_language = selected_language

        st.session_state.messages.append(
            {
                "role": "user",
                "content": task,
            }
        )

        # ---- Multi-Agent Pipeline --------------------------------
        with st.spinner(
            f"🤖 AI ASTRA multi-agent team is working on "
            f"{selected_language} code…"
        ):

            try:

                pipeline_result = run_multi_agent_pipeline(
                    llm=llm,
                    user_request=task,
                    programming_language=selected_language or "Python",
                )

                # Store activity log for sidebar display
                st.session_state.agent_activity = (
                    pipeline_result["activity_log"]
                )

                generated_code = pipeline_result["generated_code"]
                test_summary   = pipeline_result["test_results"]
                final_msg      = pipeline_result["final_response"]

                # Show the generated code in the chat
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": generated_code,
                        "type": "code",
                        "language": selected_language,
                    }
                )

                # Show the testing agent's verdict
                if test_summary:
                    verdict_icon = (
                        "✅" if "VERDICT: PASS" in test_summary.upper()
                        else "⚠️"
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                f"{verdict_icon} **Testing Agent Report**\n\n"
                                f"{test_summary}"
                            ),
                        }
                    )

                # Show orchestrator's final summary
                if final_msg:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                f"🧠 **Orchestrator Summary**\n\n{final_msg}"
                            ),
                        }
                    )

            except Exception as e:

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"❌ Multi-agent pipeline error: {str(e)}"
                        ),
                    }
                )
        # ----------------------------------------------------------

        st.session_state.pending_coding_task = None
        st.session_state.selected_language = None

        st.rerun()


# ============================================================
# CHAT INPUT
# ============================================================

if st.session_state.pending_coding_task is None:

    # Show active-file banner above the input box
    if st.session_state.uploaded_file_content is not None:
        st.info(
            f"📎 **{st.session_state.uploaded_file_name}** is loaded. "
            "Ask AI ASTRA to *analyze*, *summarize*, *explain*, "
            "*review*, or *debug* it.",
            icon="📄",
        )

    # ---- Consume a suggestion-card click as a chat message ----
    _injected = st.session_state.suggested_prompt
    if _injected:
        st.session_state.suggested_prompt = None
        user_input = _injected
    else:
        user_input = st.chat_input(
            "Ask AI ASTRA anything..."
        )

    if user_input:

        # ----------------------------------------------------
        # Show user message immediately
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        # ----------------------------------------------------
        # File analysis — if a file is loaded, use it
        # ----------------------------------------------------

        if st.session_state.uploaded_file_content is not None:

            with st.spinner(
                f"AI ASTRA is analysing "
                f"**{st.session_state.uploaded_file_name}**…"
            ):

                try:

                    answer = run_file_analysis_agent(
                        st.session_state.uploaded_file_content,
                        st.session_state.uploaded_file_name or "",
                        user_input,
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                except Exception as e:

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                f"❌ File analysis error: {str(e)}"
                            ),
                        }
                    )

            st.rerun()

        # ----------------------------------------------------
        # Detect coding request
        # ----------------------------------------------------

        with st.spinner("AI ASTRA is thinking..."):

            try:

                coding_request = is_coding_request(
                    user_input
                )

            except Exception as e:

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"❌ Classification error: {str(e)}"
                        ),
                    }
                )

                st.rerun()


        # ----------------------------------------------------
        # Coding request
        # ----------------------------------------------------

        if coding_request:

            st.session_state.pending_coding_task = (
                user_input
            )

            st.rerun()


        # ----------------------------------------------------
        # General request
        # ----------------------------------------------------

        with st.spinner(
            "AI ASTRA is working..."
        ):

            try:

                answer = run_general_agent(
                    user_input
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as e:

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"❌ Agent error: {str(e)}"
                        ),
                    }
                )

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="astra-footer" style="
        text-align:center;
        padding:25px 0 10px 0;
        font-size:12px;
    ">
        AI ASTRA • Autonomous Interlinked Agentic Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)