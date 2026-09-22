# ============================================================
# AI ASTRA — Collaborative Multi-Agent Pipeline
# ============================================================
#
# Architecture
# ─────────────
#   USER REQUEST
#        │
#        ▼
#  ORCHESTRATOR  ──decides──► needs_research? ──► RESEARCH AGENT
#        │                                              │
#        │◄──────────────────── research_result ────────┘
#        │
#        ▼
#   CODING AGENT  ──generates──► generated_code
#        ▲                              │
#        │  fix_code                    ▼
#        └──────────────────── TESTING AGENT
#                                       │
#                              bugs_found? ──► if yes + retry < 3 ──► fix loop
#                                       │
#                                       ▼  (PASS or max retries)
#                              ORCHESTRATOR (final response)
#                                       │
#                                       ▼
#                               FINAL RESPONSE
#
# ============================================================

from __future__ import annotations

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END


# ============================================================
# SHARED STATE
# ============================================================

class AgentState(TypedDict):
    """Shared state passed between all agents in the pipeline."""
    user_request:       str
    programming_language: str
    needs_research:     bool
    research_result:    str
    generated_code:     str
    test_results:       str
    bugs_found:         bool
    retry_count:        int
    final_response:     str
    # Activity log — list of human-readable status strings
    activity_log:       list[str]


MAX_RETRIES = 3


# ============================================================
# HELPERS
# ============================================================

def _log(state: AgentState, message: str) -> None:
    """Append a status message to activity_log in place."""
    state["activity_log"].append(message)


def _invoke_llm(llm, prompt: str) -> str:
    """Call the LLM and return plain string content."""
    return llm.invoke(prompt).content.strip()


# ============================================================
# NODE: ORCHESTRATOR (initial routing)
# ============================================================

def orchestrator_node(state: AgentState, llm) -> AgentState:
    """
    Analyse the user request and decide whether the Research
    Agent is needed before the Coding Agent runs.
    """
    _log(state, "🧠 Orchestrator: Analysing request…")

    prompt = f"""You are the Orchestrator of a multi-agent AI software team.

User request:
{state['user_request']}

Programming language: {state['programming_language']}

Decide whether a Research Agent should first investigate technical
approaches, algorithms, or best practices before the Coding Agent
starts writing code.

Answer with exactly one word:
YES  — if research would meaningfully improve the code quality.
NO   — if the task is straightforward and can be coded directly.
"""

    answer = _invoke_llm(llm, prompt).upper()
    state["needs_research"] = answer.startswith("YES")

    if state["needs_research"]:
        _log(state, "🧠 Orchestrator: Research needed — routing to Research Agent.")
    else:
        _log(state, "🧠 Orchestrator: No research needed — routing to Coding Agent.")

    return state


# ============================================================
# NODE: RESEARCH AGENT
# ============================================================

def research_node(state: AgentState, llm) -> AgentState:
    """
    Independently research the problem and produce structured
    technical requirements/approaches for the Coding Agent.
    """
    _log(state, "🔬 Research Agent: Investigating technical approach…")

    prompt = f"""You are an expert Research Agent in an AI software team.

Your job is to analyse the following programming task and produce
clear, structured technical guidance that the Coding Agent will use
to write the implementation.

Programming language: {state['programming_language']}

Task:
{state['user_request']}

Provide:
1. Problem breakdown — what needs to be solved.
2. Recommended algorithm / data structure / approach.
3. Key implementation steps.
4. Edge cases to handle.
5. Any relevant {state['programming_language']} libraries or built-ins.

Be concise and technical. Do NOT write code — only guidance.
"""

    result = _invoke_llm(llm, prompt)
    state["research_result"] = result
    _log(state, "🔬 Research Agent: Research complete. Passing findings to Coding Agent.")
    return state


# ============================================================
# NODE: CODING AGENT
# ============================================================

def coding_node(state: AgentState, llm) -> AgentState:
    """
    Write or fix code using the research result (if available)
    and any testing feedback from a previous iteration.
    """
    is_fix = state["retry_count"] > 0

    if is_fix:
        _log(
            state,
            f"🔧 Coding Agent: Fixing code (attempt "
            f"{state['retry_count']}/{MAX_RETRIES})…",
        )
    else:
        _log(state, "💻 Coding Agent: Generating code…")

    research_section = ""
    if state.get("research_result"):
        research_section = f"""
Research Agent findings:
{state['research_result']}

Use the above research findings to guide the implementation.
"""

    fix_section = ""
    if is_fix and state.get("test_results"):
        fix_section = f"""
Testing Agent feedback (bugs to fix):
{state['test_results']}

Fix every reported bug. Do NOT introduce new bugs.
"""

    prompt = f"""You are an expert Coding Agent in an AI software team.

Programming language: {state['programming_language']}
{research_section}
Task:
{state['user_request']}
{fix_section}

STRICT RULES:
1. Generate ONLY {state['programming_language']} code.
2. Do NOT use another language.
3. Do NOT mix languages.
4. Make the code complete and executable.
5. Include all required imports.
6. Handle input/output where needed.
7. Fix all syntax and logical errors before returning.
8. Return ONLY the final source code — no explanation, no markdown fences.
"""

    code = _invoke_llm(llm, prompt)

    # Strip accidental markdown fences
    import re
    match = re.match(
        r"^```(?:[a-zA-Z0-9_+#.-]+)?\s*(.*?)\s*```$",
        code,
        flags=re.DOTALL,
    )
    if match:
        code = match.group(1).strip()

    state["generated_code"] = code

    if is_fix:
        _log(state, "🔧 Coding Agent: Fixed code ready. Passing to Testing Agent.")
    else:
        _log(state, "💻 Coding Agent: Code generated. Passing to Testing Agent.")

    return state


# ============================================================
# NODE: TESTING AGENT
# ============================================================

def testing_node(state: AgentState, llm) -> AgentState:
    """
    Review the generated code, produce test cases, detect bugs,
    and return structured feedback.
    """
    _log(state, "🧪 Testing Agent: Reviewing code and running test analysis…")

    prompt = f"""You are an expert Testing Agent in an AI software team.

Programming language: {state['programming_language']}

Original task:
{state['user_request']}

Code to review:
```
{state['generated_code']}
```

Perform a thorough code review:
1. Syntax check — are there any syntax errors?
2. Logic check — does the code correctly solve the task?
3. Edge cases — does it handle boundary/edge conditions?
4. Runtime errors — any potential null-pointer, index, or type errors?
5. Test cases — list 2–3 concrete test cases with expected outputs.

Conclude with exactly one of:
VERDICT: PASS   — if the code is correct and handles the task properly.
VERDICT: FAIL   — if there are bugs or the code does not work correctly.

After the verdict, if FAIL, list each bug clearly so the Coding Agent can fix them.
"""

    result = _invoke_llm(llm, prompt)
    state["test_results"] = result
    state["bugs_found"] = "VERDICT: FAIL" in result.upper()

    if state["bugs_found"]:
        _log(
            state,
            f"🧪 Testing Agent: Bugs found. "
            f"Sending feedback to Coding Agent (retry "
            f"{state['retry_count'] + 1}/{MAX_RETRIES}).",
        )
    else:
        _log(state, "🧪 Testing Agent: All tests passed ✅")

    return state


# ============================================================
# NODE: ORCHESTRATOR (final response)
# ============================================================

def final_response_node(state: AgentState, llm) -> AgentState:
    """
    Compose the final user-facing response from the code and
    test results produced by the pipeline.
    """
    _log(state, "🧠 Orchestrator: Composing final response…")

    verdict = "PASS" if not state["bugs_found"] else "partial (max retries reached)"

    prompt = f"""You are the Orchestrator of an AI software team.

The team has completed the following task:
{state['user_request']}

Programming language: {state['programming_language']}

Final code produced:
```
{state['generated_code']}
```

Testing verdict: {verdict}

Testing summary:
{state['test_results']}

Write a short, professional response to the user (2–4 sentences):
- Confirm the task is complete.
- Mention the testing result.
- If there are remaining issues, name them briefly.
Do NOT repeat the code in this message.
"""

    state["final_response"] = _invoke_llm(llm, prompt)
    _log(state, "✅ Task completed.")
    return state


# ============================================================
# ROUTING FUNCTIONS
# ============================================================

def route_after_orchestrator(state: AgentState) -> str:
    return "research" if state["needs_research"] else "coding"


def route_after_testing(state: AgentState) -> str:
    if state["bugs_found"] and state["retry_count"] < MAX_RETRIES:
        state["retry_count"] += 1
        return "coding"
    return "final_response"


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_multi_agent_graph(llm):
    """
    Build and compile the multi-agent LangGraph StateGraph.
    Returns the compiled graph ready to be invoked.
    """
    from functools import partial

    # Bind LLM into each node via partial application
    _orchestrator   = partial(orchestrator_node,   llm=llm)
    _research       = partial(research_node,        llm=llm)
    _coding         = partial(coding_node,          llm=llm)
    _testing        = partial(testing_node,         llm=llm)
    _final_response = partial(final_response_node,  llm=llm)

    graph = StateGraph(AgentState)

    graph.add_node("orchestrator",    _orchestrator)
    graph.add_node("research",        _research)
    graph.add_node("coding",          _coding)
    graph.add_node("testing",         _testing)
    graph.add_node("final_response",  _final_response)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {"research": "research", "coding": "coding"},
    )

    graph.add_edge("research", "coding")

    graph.add_edge("coding", "testing")

    graph.add_conditional_edges(
        "testing",
        route_after_testing,
        {"coding": "coding", "final_response": "final_response"},
    )

    graph.add_edge("final_response", END)

    return graph.compile()


# ============================================================
# PUBLIC RUN FUNCTION
# ============================================================

def run_multi_agent_pipeline(
    llm,
    user_request: str,
    programming_language: str,
) -> AgentState:
    """
    Entry point called by the Streamlit UI.

    Returns the completed AgentState containing:
      - generated_code
      - test_results
      - final_response
      - activity_log
    """
    graph = build_multi_agent_graph(llm)

    initial_state: AgentState = {
        "user_request":        user_request,
        "programming_language": programming_language,
        "needs_research":      False,
        "research_result":     "",
        "generated_code":      "",
        "test_results":        "",
        "bugs_found":          False,
        "retry_count":         0,
        "final_response":      "",
        "activity_log":        [],
    }

    result: AgentState = graph.invoke(initial_state)
    return result
