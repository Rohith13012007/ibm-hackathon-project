# ============================================================
# AI ASTRA
# Autonomous Interlinked Agentic Intelligence System
# ============================================================

# ============================================================
# IMPORTS
# ============================================================

# Used to access environment variables
import os

# Used to call external APIs
import requests

# Load variables from .env file
from dotenv import load_dotenv

# Groq LLM integration
from langchain_groq import ChatGroq

# Structured AI tools
from langchain_core.tools import StructuredTool

# Autonomous ReAct agent
from langgraph.prebuilt import create_react_agent

# Memory system
from langgraph.checkpoint.memory import MemorySaver

# Schema validation
from pydantic import BaseModel, Field

# ============================================================
# LOAD ENV VARIABLES
# ============================================================

load_dotenv()

# ============================================================
# INITIALIZE GROQ LLM
# ============================================================

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b",
    temperature=0
)

# ============================================================
# MEMORY SYSTEM
# ============================================================

# Persistent memory across conversations
memory = MemorySaver()



# ============================================================
# RESEARCH TOOL
# ============================================================

class ResearchInput(BaseModel):

    topic: str = Field(
        description="Research topic or educational query"
    )

def research_tool_func(topic: str):

    """
    Research and explanation tool.
    """

    prompt = f"""
    Explain in detail:

    {topic}

    Include:
    - overview
    - important concepts
    - applications
    - advantages
    - future scope
    """

    response = llm.invoke(prompt)

    return response.content

# ------------------------------------------------------------
# REGISTER RESEARCH TOOL
# ------------------------------------------------------------

research_tool = StructuredTool.from_function(

    func=research_tool_func,

    name="research_tool",

    description=(

        "Use this tool for educational explanations, "
        "AI concepts, programming concepts, "
        "research topics, technical learning, "
        "and knowledge-based queries."

    ),

    args_schema=ResearchInput
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

    """
    Coding tool.
    Generates code ONLY in the selected programming language.
    """

    prompt = f"""
You are an expert software engineer.

The user selected this programming language:

{language}

Programming task:

{task}

STRICT RULES:

1. Generate code ONLY in {language}.
2. Do NOT use another programming language.
3. Do NOT mix programming languages.
4. Do NOT convert the solution to Python unless Python was selected.
5. Do NOT provide multiple language versions.
6. Keep the code simple and beginner-friendly.
7. Make the code complete and executable.
8. Check the syntax carefully.
9. If the user provided incorrect code, fix the errors.
10. If input/output is required, include it appropriately.
11. Return ONLY the {language} code.
12. Do NOT explain the code unless the user explicitly asks for an explanation.
13. Do NOT put comments outside the code.
14. Do not say "Here is the code".
"""

    response = llm.invoke(prompt)

    return response.content


# ------------------------------------------------------------
# REGISTER CODING TOOL
# ------------------------------------------------------------

coding_tool = StructuredTool.from_function(

    func=coding_tool_func,

    name="coding_tool",

    description=(

        "Use this tool for programming, coding, "
        "debugging, algorithms, APIs, software design, "
        "and development tasks. "
        "The programming language MUST already be selected "
        "before using this tool. Generate code ONLY in "
        "the selected programming language."

    ),

    args_schema=CodingInput
)

# ============================================================
# STRATEGY TOOL
# ============================================================

class StrategyInput(BaseModel):

    business_problem: str = Field(
        description="Business or startup strategy problem"
    )

def strategy_tool_func(business_problem: str):

    """
    Business strategy tool.
    """

    prompt = f"""
    You are an elite AI business strategist.

    Analyze:

    {business_problem}

    Include:
    - SWOT analysis
    - growth strategy
    - risks
    - recommendations
    """

    response = llm.invoke(prompt)

    return response.content

# ------------------------------------------------------------
# REGISTER STRATEGY TOOL
# ------------------------------------------------------------

strategy_tool = StructuredTool.from_function(

    func=strategy_tool_func,

    name="strategy_tool",

    description=(

        "Use this tool for startup ideas, "
        "business strategy, optimization, "
        "market analysis, planning, "
        "and entrepreneurial guidance."

    ),

    args_schema=StrategyInput
)

# ============================================================
# TOOL REGISTRY
# ============================================================

# ALL AI CAPABILITIES REGISTERED HERE

tools = [


    research_tool,

    coding_tool,

    strategy_tool
]

# ============================================================
# CREATE AUTONOMOUS REACT AGENT
# ============================================================

agent_executor = create_react_agent(

    llm,

    tools=tools,

    checkpointer=memory
)

# ============================================================
# TERMINAL BANNER
# ============================================================

print("\n================================================")
print("               AI ASTRA INITIALIZED")
print(" Autonomous Interlinked Agentic Intelligence")
print("================================================")

# ============================================================
# THREAD CONFIGURATION
# ============================================================

config = {
    "configurable": {
        "thread_id": "ai-astra-thread"
    }
}

# ============================================================
# CODING STATE
# ============================================================

selected_language = None
pending_coding_task = None


# ============================================================
# MAIN EXECUTION LOOP
# ============================================================

while True:

    # ========================================================
    # USER INPUT
    # ========================================================

    user_input = input("\nYou: ").strip()

    # ========================================================
    # EXIT CONDITION
    # ========================================================

    if user_input.lower() in ["exit", "quit"]:

        print("\n[AI ASTRA SHUTDOWN]")
        break


    # ========================================================
    # LANGUAGE SELECTION FOR PENDING CODING TASK
    # ========================================================

    if pending_coding_task is not None:

        language_map = {

            "1": "Python",
            "2": "Java",
            "3": "C",
            "4": "C++",
            "5": "JavaScript",
            "6": "TypeScript",
            "7": "Go",
            "8": "Rust",
            "9": "C#",

            "python": "Python",
            "java": "Java",
            "c": "C",
            "c++": "C++",
            "cpp": "C++",
            "javascript": "JavaScript",
            "js": "JavaScript",
            "typescript": "TypeScript",
            "ts": "TypeScript",
            "go": "Go",
            "golang": "Go",
            "rust": "Rust",
            "c#": "C#",
            "csharp": "C#"
        }


        selected_language = language_map.get(
            user_input.lower(),
            None
        )


        # ====================================================
        # INVALID LANGUAGE
        # ====================================================

        if selected_language is None:

            print("\nAI ASTRA: Please select a valid language.")

            print("\nAvailable languages:")
            print("1. Python")
            print("2. Java")
            print("3. C")
            print("4. C++")
            print("5. JavaScript")
            print("6. TypeScript")
            print("7. Go")
            print("8. Rust")
            print("9. C#")

            continue


        # ====================================================
        # LANGUAGE SELECTED
        # ====================================================

        print(
            f"\nAI ASTRA: Selected language → "
            f"{selected_language}"
        )

        print(
            "\nAI ASTRA: Generating code..."
        )


        # ====================================================
        # CODING AGENT
        # ====================================================

        response = agent_executor.invoke(

            {
                "messages": [

                    {
                        "role": "system",

                        "content": f"""

You are AI ASTRA Coding Agent.

The user has selected:

PROGRAMMING LANGUAGE:
{selected_language}

CODING TASK:
{pending_coding_task}

STRICT RULES:

1. Generate ONLY {selected_language} code.

2. Do NOT generate Python unless Python
   was selected.

3. Do NOT generate multiple versions.

4. Do NOT use another programming language.

5. Make the code complete.

6. Make the code executable.

7. Check syntax and fix errors.

8. Keep the code simple and beginner-friendly.

9. Include necessary imports.

10. If the task requires input, include input handling.

11. If the task requires output, include output.

12. Do NOT provide explanation.

13. Do NOT provide pseudocode.

14. Do NOT say "Here is the code".

15. Return ONLY the final {selected_language} code.

"""
                    },

                    {
                        "role": "user",

                        "content": pending_coding_task
                    }

                ]
            },

            config=config
        )


        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        print("\n================================================")
        print("FINAL RESPONSE")
        print("================================================\n")

        print(
            response["messages"][-1].content
        )


        # ====================================================
        # RESET CODING STATE
        # ====================================================

        selected_language = None
        pending_coding_task = None

        continue


    # ========================================================
    # CHECK WHETHER USER REQUEST IS CODING RELATED
    # ========================================================

    classification_prompt = f"""

Determine whether this user request is related
to programming or coding.

User request:
{user_input}

Return ONLY one of these:

CODING
GENERAL

"""

    classification_response = llm.invoke(
        classification_prompt
    )

    classification = (
        classification_response.content
        .strip()
        .upper()
    )


    # ========================================================
    # CODING REQUEST
    # ========================================================

    if classification == "CODING":

        # Save the original coding task
        pending_coding_task = user_input

        print("\n================================================")
        print("AI ASTRA - PROGRAMMING LANGUAGE")
        print("================================================")

        print(
            "\nWhich programming language would you like?"
        )

        print("\n1. Python")
        print("2. Java")
        print("3. C")
        print("4. C++")
        print("5. JavaScript")
        print("6. TypeScript")
        print("7. Go")
        print("8. Rust")
        print("9. C#")

        continue


    # ========================================================
    # NORMAL AI ASTRA AGENT
    # ========================================================

    response = agent_executor.invoke(

        {
            "messages": [

                {
                    "role": "system",

                    "content": """

You are AI ASTRA.

An advanced autonomous interlinked
agentic intelligence system.

Capabilities:

- autonomous reasoning
- intelligent tool usage
- memory-aware conversation
- coding assistance
- business strategy
- technical research

Rules:

- Decide tools autonomously.
- Use tools whenever required.
- Use multiple tools if necessary.
- Maintain conversational memory.
- Give professional responses.
- Never expose internal chain-of-thought.
- Never expose hidden reasoning.

For coding requests:

- Ask the user for the programming language first.
- Never generate code before the language is selected.
- Generate code only in the selected language.

"""
                },

                {
                    "role": "user",

                    "content": user_input
                }

            ]
        },

        config=config
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n================================================")
    print("FINAL RESPONSE")
    print("================================================\n")

    print(
        response["messages"][-1].content
    )