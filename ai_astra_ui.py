# ============================================================
# AI ASTRA
# Autonomous Interlinked Agentic Intelligence System
# Streamlit UI Version
# ============================================================

import os
import re

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field


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

st.markdown(
    """
    <style>
        .stApp {
            background: #0b1020;
        }

        [data-testid="stSidebar"] {
            background: #111827;
        }

        .astra-title {
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 0;
        }

        .astra-subtitle {
            color: #9ca3af;
            font-size: 16px;
            margin-top: 0;
        }

        .status-card {
            padding: 14px 18px;
            border-radius: 12px;
            background: #111827;
            border: 1px solid #263244;
            margin-bottom: 15px;
        }

        .language-card {
            padding: 18px;
            border-radius: 14px;
            background: #111827;
            border: 1px solid #334155;
            margin-top: 10px;
            margin-bottom: 15px;
        }

        .small-text {
            color: #9ca3af;
            font-size: 13px;
        }

        .stChatMessage {
            border-radius: 14px;
        }

        code {
            font-size: 14px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    Run the normal AI ASTRA ReAct agent.
    """

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
- Maintain conversational memory.
- Give professional, clear answers.
- Never expose chain-of-thought.
- Never expose hidden reasoning.

IMPORTANT:
Coding requests are handled by the application's
programming-language selection flow. If a coding request
reaches you directly, ask for the programming language
instead of generating code.
""",
                },
                {
                    "role": "user",
                    "content": user_input,
                },
            ]
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
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🤖 AI ASTRA")

    st.markdown(
        """
        **Autonomous Interlinked Agentic Intelligence**

        AI capabilities:
        - 🔬 Research
        - 💻 Coding
        - 📊 Strategy
        - 🧠 Memory
        - 🔗 ReAct Agent
        """
    )

    st.divider()

    st.markdown("### Agent Status")

    st.success("ONLINE")

    st.caption(
        "Model: openai/gpt-oss-120b"
    )

    st.caption(
        "Framework: LangGraph + LangChain"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.pending_coding_task = None
        st.session_state.selected_language = None
        st.session_state.thread_id = (
            f"ai-astra-streamlit-{id(st.session_state)}"
        )
        st.rerun()


# ============================================================
# HEADER
# ============================================================

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

        with st.spinner(
            f"AI ASTRA is generating {selected_language} code..."
        ):

            try:

                code = run_coding_agent(
                    task,
                    selected_language,
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": code,
                        "type": "code",
                        "language": selected_language,
                    }
                )

            except Exception as e:

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"❌ Coding error: {str(e)}"
                        ),
                    }
                )

        st.session_state.pending_coding_task = None
        st.session_state.selected_language = None

        st.rerun()


# ============================================================
# CHAT INPUT
# ============================================================

if st.session_state.pending_coding_task is None:

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
    """
    <div style="
        text-align:center;
        color:#6b7280;
        padding:25px 0 10px 0;
        font-size:12px;
    ">
        AI ASTRA • Autonomous Interlinked Agentic Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)