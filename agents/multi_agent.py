# ============================================================
# AI ASTRA — Collaborative Multi-Agent Pipeline
# ============================================================
#
# Architecture
# ─────────────
#
#   USER REQUEST
#        │
#        ▼
#   SUPERVISOR / ORCHESTRATOR
#        │
#        ▼
#   REQUIREMENTS SPECIFICATION AGENT
#        │  (generates structured spec + REQ-IDs)
#        ▼
#   RESEARCH AGENT  (optional, decided by orchestrator)
#        │
#        ▼
#   CODING AGENT
#        │
#        ▼
#   TESTING AGENT
#        │
#   Tests failed?
#   YES ─────────────► DEBUGGING AGENT
#   │                        │
#   │                   CODING AGENT
#   │                        │
#   │                   TESTING AGENT ──► (loop, max MAX_RETRIES)
#   │
#   NO
#   │
#   ▼
#   REVIEW AGENT
#        │
#   Issues (CRITICAL/HIGH)?
#   YES ─────────────► DEBUGGING AGENT
#   │                        │
#   │                   CODING AGENT
#   │                        │
#   │                   TESTING AGENT
#   │                        │
#   │                   REVIEW AGENT ──► (loop, max MAX_REVIEW_RETRIES)
#   │
#   NO
#   │
#   ▼
#   FINAL RESPONSE
#
# ============================================================

from __future__ import annotations

import json
import re
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END


# ============================================================
# SHARED STATE
# ============================================================

class AgentState(TypedDict):
    """Shared state passed between all agents in the pipeline."""
    # ── Core request ──────────────────────────────────────────
    user_request:                   str
    programming_language:           str

    # ── Orchestrator routing ─────────────────────────────────
    needs_research:                 bool

    # ── Requirements Specification Agent ─────────────────────
    requirements_specification:     dict        # structured JSON spec
    requirements_ids:               list[str]   # ["REQ-001", "REQ-002", …]
    ambiguities:                    list[str]   # unresolved ambiguities
    acceptance_criteria:            list[str]   # traceability criteria

    # ── Research Agent ────────────────────────────────────────
    research_result:                str

    # ── Coding Agent ─────────────────────────────────────────
    generated_code:                 str

    # ── Testing Agent ─────────────────────────────────────────
    test_results:                   str
    bugs_found:                     bool
    retry_count:                    int         # debug-retry counter

    # ── Debugging Agent ───────────────────────────────────────
    debug_report:                   str

    # ── Review Agent ─────────────────────────────────────────
    review_result:                  str         # full structured review text
    review_status:                  str         # "PASS" | "NEEDS_IMPROVEMENT"
    requirements_coverage:          dict        # {implemented, partial, missing}
    review_findings:                list[str]   # CRITICAL/HIGH/MEDIUM findings
    critical_issues:                list[str]   # CRITICAL issues only
    review_iteration:               int         # review-retry counter

    # ── Pipeline control ─────────────────────────────────────
    final_response:                 str
    activity_log:                   list[str]   # human-readable status log


MAX_RETRIES        = 3   # max debug-fix loops after test failures
MAX_REVIEW_RETRIES = 3   # max review-fix loops after review findings


# ============================================================
# HELPERS
# ============================================================

def _log(state: AgentState, message: str) -> None:
    """Append a status message to activity_log in place."""
    state["activity_log"].append(message)


def _invoke_llm(llm, prompt: str) -> str:
    """Call the LLM and return plain string content."""
    return llm.invoke(prompt).content.strip()


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    match = re.match(
        r"^```(?:[a-zA-Z0-9_+#.-]+)?\s*(.*?)\s*```$",
        text,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else text.strip()


# ============================================================
# NODE: ORCHESTRATOR (initial routing)
# ============================================================

def orchestrator_node(state: AgentState, llm) -> AgentState:
    """
    Analyse the user request and decide whether the Research
    Agent is needed before the Coding Agent runs.
    """
    _log(state, "🧠 Supervisor: Analysing request and routing…")

    prompt = f"""You are the Supervisor Orchestrator of a multi-agent AI software team.

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
        _log(state, "🧠 Supervisor: Research needed — routing to Requirements → Research → Coding.")
    else:
        _log(state, "🧠 Supervisor: No research needed — routing to Requirements → Coding.")

    return state


# ============================================================
# NODE: REQUIREMENTS SPECIFICATION AGENT
# ============================================================

def requirements_node(state: AgentState, llm) -> AgentState:
    """
    Convert the user's natural-language request into a clear,
    structured software specification with numbered REQ-IDs.
    Identifies ambiguities without inventing requirements.
    """
    _log(state, "📋 Requirements Agent: Analysing user request and generating specification…")

    prompt = f"""You are an expert Requirements Specification Agent in an AI software team.

Your responsibility is to convert the user's natural-language request into a
clear, structured software specification.

Programming language: {state['programming_language']}

User request:
{state['user_request']}

Produce a JSON object with EXACTLY this structure (no additional text, no markdown fences):

{{
  "project_title": "Short title for the project",
  "summary": "One-paragraph project summary",
  "functional_requirements": [
    "REQ-001: Description",
    "REQ-002: Description"
  ],
  "non_functional_requirements": [
    "Performance: ...",
    "Security: ...",
    "Maintainability: ..."
  ],
  "user_roles": ["Role 1", "Role 2"],
  "inputs": ["Input 1", "Input 2"],
  "outputs": ["Output 1", "Output 2"],
  "constraints": ["Constraint 1"],
  "dependencies": ["Dependency 1"],
  "edge_cases": ["Edge case 1", "Edge case 2"],
  "acceptance_criteria": [
    "REQ-001: User can ...",
    "REQ-002: System must ..."
  ],
  "ambiguities": [
    "Authentication method not specified",
    "Database not specified"
  ],
  "priority_requirements": ["REQ-001", "REQ-002"]
}}

RULES:
- Assign REQ-IDs starting from REQ-001 to every functional requirement.
- Do NOT invent requirements beyond what the user asked.
- Clearly list any ambiguities (missing or underspecified details).
- If the user explicitly mentions a technology, include it in dependencies.
- Return ONLY the JSON — no surrounding text.
"""

    raw = _invoke_llm(llm, prompt)

    # ── Parse JSON, fall back to a minimal safe structure ──────────
    try:
        # Strip any accidental fences
        cleaned = _strip_fences(raw)
        # Remove control characters that can break JSON parsing
        cleaned = re.sub(r'[\x00-\x1f\x7f]', lambda m: ' ' if m.group() in ('\n', '\r', '\t') else '', cleaned)
        spec = json.loads(cleaned)
    except Exception:
        # Build a minimal spec so the pipeline can continue
        spec = {
            "project_title":               "Software Project",
            "summary":                     state["user_request"],
            "functional_requirements":     ["REQ-001: Implement requested functionality"],
            "non_functional_requirements": ["Performance: Reasonable response time"],
            "user_roles":                  ["User"],
            "inputs":                      ["User input"],
            "outputs":                     ["System output"],
            "constraints":                 [],
            "dependencies":                [state["programming_language"]],
            "edge_cases":                  ["Invalid input"],
            "acceptance_criteria":         ["REQ-001: Functionality works correctly"],
            "ambiguities":                 ["Requirements could not be fully parsed from LLM response"],
            "priority_requirements":       ["REQ-001"],
        }

    state["requirements_specification"] = spec

    # ── Extract REQ-IDs ─────────────────────────────────────────────
    req_ids: list[str] = []
    for item in spec.get("functional_requirements", []):
        m = re.match(r"(REQ-\d+)", str(item))
        if m:
            req_ids.append(m.group(1))
    state["requirements_ids"]    = req_ids
    state["ambiguities"]         = spec.get("ambiguities", [])
    state["acceptance_criteria"] = spec.get("acceptance_criteria", [])

    title = spec.get("project_title", "Project")
    n_reqs = len(req_ids)
    n_amb  = len(state["ambiguities"])

    _log(
        state,
        f"📋 Requirements Agent: Specification complete — "
        f"{n_reqs} requirements identified"
        + (f", {n_amb} ambiguit{'y' if n_amb == 1 else 'ies'} noted." if n_amb else "."),
    )
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

    spec = state.get("requirements_specification", {})
    spec_summary = ""
    if spec:
        spec_summary = f"""
Requirements Specification:
Project: {spec.get('project_title', '')}
Summary: {spec.get('summary', '')}
Functional requirements: {', '.join(spec.get('functional_requirements', []))}
Dependencies: {', '.join(spec.get('dependencies', []))}
"""

    prompt = f"""You are an expert Research Agent in an AI software team.

Your job is to analyse the following programming task and produce
clear, structured technical guidance that the Coding Agent will use
to write the implementation.

Programming language: {state['programming_language']}
{spec_summary}
Original task:
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
    and any testing / debugging feedback from a previous iteration.
    """
    is_fix = state["retry_count"] > 0 or state["review_iteration"] > 0

    if is_fix:
        iteration = state["retry_count"] + state["review_iteration"]
        _log(
            state,
            f"💻 Coding Agent: Applying fix (iteration {iteration})…",
        )
    else:
        _log(state, "💻 Coding Agent: Generating code…")

    # ── Build context sections ────────────────────────────────
    spec = state.get("requirements_specification", {})
    spec_section = ""
    if spec:
        reqs = "\n".join(
            f"  {r}" for r in spec.get("functional_requirements", [])
        )
        criteria = "\n".join(
            f"  {c}" for c in spec.get("acceptance_criteria", [])
        )
        spec_section = f"""
Requirements Specification:
Project: {spec.get('project_title', '')}
Summary: {spec.get('summary', '')}

Functional Requirements:
{reqs}

Acceptance Criteria:
{criteria}
"""

    research_section = ""
    if state.get("research_result"):
        research_section = f"""
Research Agent findings:
{state['research_result']}

Use the above research findings to guide the implementation.
"""

    fix_section = ""
    if is_fix:
        if state.get("debug_report"):
            fix_section = f"""
Debugging Agent fix guidance:
{state['debug_report']}

Apply the above fix precisely.
"""
        elif state.get("test_results"):
            fix_section = f"""
Testing Agent feedback (bugs to fix):
{state['test_results']}

Fix every reported bug. Do NOT introduce new bugs.
"""

    prompt = f"""You are an expert Coding Agent in an AI software team.

Programming language: {state['programming_language']}
{spec_section}{research_section}
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
    code = _strip_fences(code)
    state["generated_code"] = code

    if is_fix:
        _log(state, "💻 Coding Agent: Fix applied. Passing to Testing Agent.")
    else:
        _log(state, "💻 Coding Agent: Code generated. Passing to Testing Agent.")

    return state


# ============================================================
# NODE: TESTING AGENT
# ============================================================

def testing_node(state: AgentState, llm) -> AgentState:
    """
    Review the generated code, produce test cases, detect bugs,
    and return structured feedback.  Uses acceptance criteria for
    requirements traceability where available.
    """
    _log(state, "🧪 Testing Agent: Reviewing code and running test analysis…")

    # Build acceptance criteria traceability block
    criteria = state.get("acceptance_criteria", [])
    criteria_section = ""
    if criteria:
        criteria_section = "\nAcceptance criteria to verify:\n" + "\n".join(
            f"  {c}" for c in criteria
        ) + "\n"

    prompt = f"""You are an expert Testing Agent in an AI software team.

Programming language: {state['programming_language']}

Original task:
{state['user_request']}
{criteria_section}
Code to review:
```
{state['generated_code']}
```

Perform a thorough code review:
1. Syntax check — are there any syntax errors?
2. Logic check — does the code correctly solve the task?
3. Edge cases — does it handle boundary/edge conditions?
4. Runtime errors — any potential null-pointer, index, or type errors?
5. Acceptance criteria — does the implementation satisfy each criterion above?
6. Test cases — list 2–3 concrete test cases with expected outputs.

Conclude with exactly one of:
VERDICT: PASS   — if the code is correct and handles the task properly.
VERDICT: FAIL   — if there are bugs or the code does not work correctly.

After the verdict, if FAIL, list each bug clearly so the Debugging Agent can fix them.
"""

    result = _invoke_llm(llm, prompt)
    state["test_results"] = result
    state["bugs_found"]   = "VERDICT: FAIL" in result.upper()

    if state["bugs_found"]:
        _log(
            state,
            f"🧪 Testing Agent: Bugs found ⚠️ — "
            f"routing to Debugging Agent (retry {state['retry_count'] + 1}/{MAX_RETRIES}).",
        )
    else:
        _log(state, "🧪 Testing Agent: All tests passed ✅")

    return state


# ============================================================
# NODE: DEBUGGING AGENT
# ============================================================

def debugging_node(state: AgentState, llm) -> AgentState:
    """
    Analyse the failed code + test results (or review findings),
    identify the root cause, and produce a patched version.
    The patched code is written back to state['generated_code'].
    """
    iteration = state["retry_count"] + state["review_iteration"]
    _log(
        state,
        f"🐛 Debugging Agent: Analysing errors (iteration {iteration})…",
    )

    research_ctx = ""
    if state.get("research_result"):
        research_ctx = f"""
Relevant Research Agent findings:
{state['research_result']}
"""

    # Use review findings if this debug run comes from Review Agent
    error_context = state.get("test_results", "")
    if state.get("review_findings"):
        review_issues = "\n".join(state["review_findings"])
        error_context = (
            f"Review Agent issues:\n{review_issues}\n\n"
            f"Testing Agent report:\n{error_context}"
        )

    prompt = f"""You are an expert Debugging Agent in an AI software team.

Your sole responsibility is to identify bugs, explain them clearly,
and produce corrected code.

Programming language: {state['programming_language']}

Original task:
{state['user_request']}
{research_ctx}
--- CODE UNDER REVIEW ---
{state['generated_code']}
--- END CODE ---

--- ERROR CONTEXT ---
{error_context}
--- END CONTEXT ---

Perform a thorough debugging analysis and respond with the following
EXACT sections (use these headers verbatim):

## Error Identification
List every distinct error found (syntax, runtime, logic, type, etc.).

## Root Cause
Explain WHY each error occurs.

## Affected Location
For each error, state the function / line / block that contains it.

## Suggested Fix
Describe the fix in plain English before writing any code.

## Corrected Code
Provide the COMPLETE corrected source file — no omissions, no placeholders.
Return ONLY valid {state['programming_language']} code in this section,
without markdown fences.

## Fix Explanation
Briefly explain what was changed and why the fix resolves the issue.

## Verification Recommendation
Suggest how to verify the fix (test cases, edge cases, manual checks).
"""

    debug_output = _invoke_llm(llm, prompt)

    # ── Extract the corrected code section ──────────────────────────
    corrected_match = re.search(
        r"##\s*Corrected Code\s*\n(.*?)(?=\n##\s|\Z)",
        debug_output,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if corrected_match:
        patched = corrected_match.group(1).strip()
        patched = _strip_fences(patched)
        if patched:
            state["generated_code"] = patched

    state["debug_report"] = debug_output
    _log(
        state,
        "🐛 Debugging Agent: Debug report ready — "
        "patched code passing to Coding Agent for verification.",
    )
    return state


# ============================================================
# NODE: REVIEW AGENT
# ============================================================

def review_node(state: AgentState, llm) -> AgentState:
    """
    Review the complete implementation against the Requirements
    Specification, test results, code quality, security,
    architecture, and every REQ-ID.
    """
    _log(state, "🔎 Review Agent: Reviewing implementation against requirements…")

    spec = state.get("requirements_specification", {})
    req_list = "\n".join(
        f"  {r}" for r in spec.get("functional_requirements", [])
    )
    nfr_list = "\n".join(
        f"  {n}" for n in spec.get("non_functional_requirements", [])
    )
    criteria_list = "\n".join(
        f"  {c}" for c in spec.get("acceptance_criteria", [])
    )
    edge_cases = "\n".join(
        f"  {e}" for e in spec.get("edge_cases", [])
    )

    prompt = f"""You are an expert Review Agent in an AI software team.

Your responsibility is to review the complete implementation against
the Requirements Specification, test results, code quality, security,
architecture, edge cases, and requirements traceability.

Programming language: {state['programming_language']}

=== REQUIREMENTS SPECIFICATION ===
Project: {spec.get('project_title', 'N/A')}
Summary: {spec.get('summary', 'N/A')}

Functional Requirements:
{req_list or '  (none specified)'}

Non-Functional Requirements:
{nfr_list or '  (none specified)'}

Acceptance Criteria:
{criteria_list or '  (none specified)'}

Edge Cases:
{edge_cases or '  (none specified)'}

=== GENERATED CODE ===
```
{state['generated_code']}
```

=== TESTING AGENT REPORT ===
{state['test_results']}

=== DEBUGGING REPORT (if any) ===
{state.get('debug_report', 'No debugging was performed.')}

=== REVIEW TASK ===

Produce a structured review with these EXACT sections:

## Requirements Coverage
For EACH REQ-ID listed above state:
  REQ-XXX: [Implemented / Partially Implemented / Missing] — brief evidence

## Code Quality
List findings (naming, structure, duplication, complexity).

## Security Findings
List any security concerns (injection, auth, validation, etc.).

## Architecture Findings
List structural or design concerns.

## Testing Coverage
Does the test analysis cover all acceptance criteria?

## Bugs Found
List any remaining bugs NOT addressed by debugging.

## Improvements
List recommended improvements (not blockers).

## Critical Issues
List ONLY genuinely critical implementation problems using this format:
  CRITICAL: <description>
  HIGH: <description>
  MEDIUM: <description>
  LOW: <description>
  INFO: <description>

## Review Verdict
End with exactly one of:
REVIEW: PASS           — all requirements met, no critical issues
REVIEW: NEEDS_IMPROVEMENT — one or more CRITICAL or HIGH issues found

Do NOT claim code passed tests unless the Testing Agent reports VERDICT: PASS.
Do NOT claim code was executed if it was not executed.
"""

    review_output = _invoke_llm(llm, prompt)
    state["review_result"] = review_output

    # ── Parse review verdict ────────────────────────────────────────
    if "REVIEW: PASS" in review_output.upper():
        state["review_status"] = "PASS"
    else:
        state["review_status"] = "NEEDS_IMPROVEMENT"

    # ── Extract CRITICAL / HIGH findings ───────────────────────────
    critical: list[str] = []
    findings: list[str] = []
    for line in review_output.splitlines():
        stripped = line.strip()
        if re.match(r"^(CRITICAL|HIGH):", stripped, re.IGNORECASE):
            critical.append(stripped)
            findings.append(stripped)
        elif re.match(r"^(MEDIUM|LOW|INFO):", stripped, re.IGNORECASE):
            findings.append(stripped)

    state["critical_issues"]  = critical
    state["review_findings"]  = findings

    # ── Build requirements coverage summary ─────────────────────────
    implemented: list[str] = []
    partial: list[str]     = []
    missing: list[str]     = []
    for line in review_output.splitlines():
        m = re.match(r"(REQ-\d+):\s*(Implemented|Partially Implemented|Missing)", line, re.IGNORECASE)
        if m:
            req_id, status = m.group(1), m.group(2).lower()
            if "partial" in status:
                partial.append(req_id)
            elif "missing" in status:
                missing.append(req_id)
            else:
                implemented.append(req_id)

    state["requirements_coverage"] = {
        "implemented": implemented,
        "partially_implemented": partial,
        "missing": missing,
    }

    if state["review_status"] == "PASS":
        _log(state, "🔎 Review Agent: Review PASSED ✅ — all requirements satisfied.")
    else:
        n_crit = len(critical)
        _log(
            state,
            f"🔎 Review Agent: Review NEEDS_IMPROVEMENT ⚠️ — "
            f"{n_crit} critical/high issue{'s' if n_crit != 1 else ''} found.",
        )

    return state


# ============================================================
# NODE: ORCHESTRATOR (final response)
# ============================================================

def final_response_node(state: AgentState, llm) -> AgentState:
    """
    Compose the final user-facing response from the code,
    test results, and review outcome.
    """
    _log(state, "🧠 Supervisor: Composing final response…")

    test_verdict  = "PASS" if not state["bugs_found"] else "partial (max retries reached)"
    review_verdict = state.get("review_status", "N/A")

    spec = state.get("requirements_specification", {})
    reqs_covered = state.get("requirements_coverage", {})

    prompt = f"""You are the Supervisor Orchestrator of an AI software team.

The team has completed the following task:
{state['user_request']}

Programming language: {state['programming_language']}
Project: {spec.get('project_title', 'N/A')}

Final code produced:
```
{state['generated_code']}
```

Testing verdict: {test_verdict}
Review verdict: {review_verdict}

Requirements coverage:
  Implemented: {', '.join(reqs_covered.get('implemented', [])) or 'none'}
  Partially implemented: {', '.join(reqs_covered.get('partially_implemented', [])) or 'none'}
  Missing: {', '.join(reqs_covered.get('missing', [])) or 'none'}

Write a short, professional response to the user (3–5 sentences):
- Confirm the task is complete.
- Mention the testing and review results.
- State which requirements were fully implemented.
- If there are remaining issues, name them briefly.
Do NOT repeat the code in this message.
"""

    state["final_response"] = _invoke_llm(llm, prompt)
    _log(state, "✅ Pipeline complete — final response ready.")
    return state


# ============================================================
# ROUTING FUNCTIONS
# ============================================================

def route_after_orchestrator(state: AgentState) -> str:
    """Always route to requirements first."""
    return "requirements"


def route_after_requirements(state: AgentState) -> str:
    """After requirements, decide research or coding."""
    return "research" if state["needs_research"] else "coding"


def route_after_testing(state: AgentState) -> str:
    """
    Tests failed + retries left  → Debugging Agent.
    Tests passed                 → Review Agent.
    Tests failed + max retries   → Review Agent anyway (best effort).
    """
    if state["bugs_found"] and state["retry_count"] < MAX_RETRIES:
        return "debugging"
    return "review"


def route_after_debugging(state: AgentState) -> str:
    """
    After debugging, always go back to Coding Agent and increment counter.
    Determines which counter to increment based on review_iteration.
    """
    if state["review_iteration"] > 0:
        state["review_iteration"] += 1
    else:
        state["retry_count"] += 1
    return "coding"


def route_after_review(state: AgentState) -> str:
    """
    Review PASS                         → final response.
    Review NEEDS_IMPROVEMENT + retries  → debugging loop.
    Review NEEDS_IMPROVEMENT + max      → final response (best effort).
    """
    if (
        state["review_status"] == "NEEDS_IMPROVEMENT"
        and state["review_iteration"] < MAX_REVIEW_RETRIES
    ):
        # Bump review iteration so debugging_node knows its context
        state["review_iteration"] += 1
        return "debugging"
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
    _orchestrator    = partial(orchestrator_node,    llm=llm)
    _requirements    = partial(requirements_node,    llm=llm)
    _research        = partial(research_node,        llm=llm)
    _coding          = partial(coding_node,          llm=llm)
    _testing         = partial(testing_node,         llm=llm)
    _debugging       = partial(debugging_node,       llm=llm)
    _review          = partial(review_node,          llm=llm)
    _final_response  = partial(final_response_node,  llm=llm)

    graph = StateGraph(AgentState)

    graph.add_node("orchestrator",   _orchestrator)
    graph.add_node("requirements",   _requirements)
    graph.add_node("research",       _research)
    graph.add_node("coding",         _coding)
    graph.add_node("testing",        _testing)
    graph.add_node("debugging",      _debugging)
    graph.add_node("review",         _review)
    graph.add_node("final_response", _final_response)

    # ── Entry point ──────────────────────────────────────────────────
    graph.set_entry_point("orchestrator")

    # orchestrator → requirements (always)
    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {"requirements": "requirements"},
    )

    # requirements → research OR coding
    graph.add_conditional_edges(
        "requirements",
        route_after_requirements,
        {"research": "research", "coding": "coding"},
    )

    # research → coding (always)
    graph.add_edge("research", "coding")

    # coding → testing (always)
    graph.add_edge("coding", "testing")

    # testing → debugging OR review
    graph.add_conditional_edges(
        "testing",
        route_after_testing,
        {"debugging": "debugging", "review": "review"},
    )

    # debugging → coding (always, counter incremented inside router)
    graph.add_conditional_edges(
        "debugging",
        route_after_debugging,
        {"coding": "coding"},
    )

    # review → debugging OR final_response
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {"debugging": "debugging", "final_response": "final_response"},
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
      - requirements_specification  (structured spec dict)
      - requirements_ids            (["REQ-001", …])
      - ambiguities                 (list of strings)
      - acceptance_criteria         (list of strings)
      - research_result
      - generated_code
      - test_results
      - debug_report
      - review_result
      - review_status               ("PASS" | "NEEDS_IMPROVEMENT")
      - requirements_coverage       ({implemented, partial, missing})
      - review_findings
      - critical_issues
      - final_response
      - activity_log
    """
    graph = build_multi_agent_graph(llm)

    initial_state: AgentState = {
        "user_request":               user_request,
        "programming_language":       programming_language,
        "needs_research":             False,
        # Requirements
        "requirements_specification": {},
        "requirements_ids":           [],
        "ambiguities":                [],
        "acceptance_criteria":        [],
        # Research
        "research_result":            "",
        # Coding
        "generated_code":             "",
        # Testing
        "test_results":               "",
        "bugs_found":                 False,
        "retry_count":                0,
        # Debugging
        "debug_report":               "",
        # Review
        "review_result":              "",
        "review_status":              "",
        "requirements_coverage":      {"implemented": [], "partially_implemented": [], "missing": []},
        "review_findings":            [],
        "critical_issues":            [],
        "review_iteration":           0,
        # Pipeline
        "final_response":             "",
        "activity_log":               [],
    }

    result: AgentState = graph.invoke(initial_state)
    return result
