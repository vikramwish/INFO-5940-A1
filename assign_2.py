# app.py
"""
Multi-Agent Travel Planner

Highlights:
- Clear separation of concerns (tools, agents, orchestration, UI)
- Simple global logger to display tool calls live in the sidebar
- Planner → Reviewer pipeline enforced before rendering any answer
- Minimal dependencies and straightforward control flow

"""

from __future__ import annotations

import os
import asyncio
import time
from typing import Callable, Dict, List, Optional, Any

import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient
import altair as alt
import graphviz
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Environment & Globals
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv(override=True)  # Loads variables from a local .env if present
os.environ.setdefault("OPENAI_LOG", "error")
os.environ.setdefault("OPENAI_TRACING", "false")

# Tool call logger: the UI sets this per request. The tool checks it and logs.
# Using a simple global makes this easy to teach and reason about.
TOOL_LOGGER: Optional[Callable[[Dict[str, Any]], None]] = None
TOOL_LOGS: List[str] = []


def set_tool_logger(logger: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Install or remove the UI logger used by tools to report activity."""
    global TOOL_LOGGER
    TOOL_LOGGER = logger


def log_tool_event(event: Dict[str, Any]) -> None:
    """If a logger is installed, send the event to the UI."""
    if TOOL_LOGGER is not None:
        try:
            TOOL_LOGGER(event)
        except Exception:
            # Logging should never break the app or the tool itself
            pass
    # Append to global logs
    TOOL_LOGS.append(str(event))


def redact_for_logs(value: Any) -> Any:
    """
    Make sure we don't leak secrets and keep logs small.
    This is deliberately simple for teaching.
    """
    if isinstance(value, str):
        low = value.lower()
        if any(k in low for k in ("api_key", "token", "secret", "password")):
            return "[redacted]"
        return value if len(value) <= 300 else value[:120] + "… [truncated]"
    if isinstance(value, dict):
        return {k: ("[redacted]" if any(s in k.lower() for s in ("key", "token", "secret", "password"))
                    else redact_for_logs(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact_for_logs(v) for v in value]
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def parse_currency(text: str) -> list[dict]:
    """
    Parse currency amounts from text, e.g., ~$40, $20.
    Returns list of dicts with 'amount': float, 'description': str.
    """
    import re
    matches = re.findall(r'~\$?(\d+(?:\.\d+)?)', text)
    return [{'amount': float(m), 'description': f'Cost: ${m}'} for m in matches]


def split_days(markdown: str) -> list[dict]:
    """
    Split Markdown itinerary into day sections.
    Returns list of dicts with 'title': str, 'content': str.
    """
    import re
    days = re.split(r'(?=\*\*Day \d+)', markdown)
    result = []
    for day in days:
        if day.strip():
            lines = day.strip().split('\n')
            title = lines[0] if lines else 'Unknown'
            content = '\n'.join(lines[1:]) if len(lines) > 1 else ''
            result.append({'title': title, 'content': content})
    return result


def render_cost_chart(costs: list[dict]) -> alt.Chart:
    """
    Render a bar chart of costs using Altair.
    """
    if not costs:
        return alt.Chart().mark_text(text="No costs found").encode()
    df = pd.DataFrame(costs)
    chart = alt.Chart(df).mark_bar().encode(
        x='description:N',
        y='amount:Q',
        color='description:N'
    ).properties(title="Estimated Costs")
    return chart


def render_flow_diagram() -> str:
    """
    Return Graphviz DOT string for the agent flow.
    """
    return """
    digraph {
        rankdir=LR;
        User [shape=box];
        Planner [shape=box];
        Reviewer [shape=box];
        Output [shape=box];
        User -> Planner [label="Prompt"];
        Planner -> Reviewer [label="Draft"];
        Reviewer -> Output [label="Validated"];
    }
    """


# ──────────────────────────────────────────────────────────────────────────────
# Agent Framework Imports (provided by you)
# ──────────────────────────────────────────────────────────────────────────────
# These come from your own framework. We assume:
# - Agent: defines a model + instructions + optional tools
# - Runner.run(agent, input): executes an agent and returns an object with text
from agents import Agent, Runner, function_tool  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────

@function_tool
def internet_search(query: str) -> str:
    """
    Internet search backed by Tavily.
    - Reads TAVILY_API_KEY from environment.
    - Sends simple log events before/after the call so the UI can show activity.
    """
    log_tool_event({"type": "call", "tool": "internet_search", "args": {"query": redact_for_logs(query)}})

    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            msg = "missing TAVILY_API_KEY in environment."
            log_tool_event({"type": "error", "tool": "internet_search", "error": msg})
            return f"Search error: {msg}"

        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=3)

        items = response.get("results", [])
        lines = [f"- {it.get('title', 'N/A')}: {it.get('content', 'N/A')}" for it in items]
        output = "\n".join(lines) if lines else "No results found."

        log_tool_event({
            "type": "result",
            "tool": "internet_search",
            "preview": redact_for_logs(output[:400] + ("…" if len(output) > 400 else "")),
        })
        return output

    except Exception as e:
        log_tool_event({"type": "error", "tool": "internet_search", "error": str(e)})
        return f"Search error: {e}"

    finally:
        log_tool_event({"type": "end", "tool": "internet_search"})


# ──────────────────────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────────────────────

# BEGIN SOLUTION
REVIEWER_INSTRUCTIONS = """
You are the Reviewer Agent for a multi-agent travel assistant.

Your role:
- Validate and fact-check the itinerary created by the Planner Agent.

Your process:
1. Review the proposed itinerary for:
   - Feasibility (e.g., opening hours, realistic travel times, ticket prices, activity durations).
   - Consistency with user constraints such as budget, time, and interests.
   - Logical sequencing of activities and cities.
2. Use the `internet_search` tool for real-time verification and fact-checking.
3. Identify any errors, unrealistic assumptions, or missing information.
4. Suggest specific improvements in a structured 'Delta List' format.

Output format:
---
### Review Summary
- Overall impression of itinerary quality and feasibility.

### Delta List (Suggested Fixes)
1. [Issue] — [Reason] — [Proposed Change]
2. ...
---

Objective:
Ensure the final itinerary is realistic, verifiable, and user-aligned before returning it to the user.
"""

PLANNER_INSTRUCTIONS = """
You are the Planner Agent for a multi-agent travel assistant.

Goal:
Transform a vague travel request into a clear, day-by-day itinerary that feels realistic, balanced, and aligned with the user’s preferences.

Your tasks:
1. Carefully interpret the user’s travel prompt (budget, duration, interests, locations, pacing).
2. Generate a day-by-day itinerary with:
   - Morning, afternoon, and evening activities (include times and brief details).
   - Approximate costs for each activity and per-day totals.
   - Logical city progression and transport time between places.
   - Notes on lodging, food, and travel logistics.
3. Stay within the budget and timeline constraints mentioned.
4. Work entirely from your own knowledge (no internet access).
5. Present the itinerary in a structured Markdown format that’s visually clear and easy to read.

Output format example:
---
**Day 1 – Arrival in Rome**
- Morning: Check-in at hostel near Termini Station (~$40)
- Afternoon: Visit the Colosseum (2:00–5:00 PM, ~$20)
- Evening: Dinner in Trastevere (~$25)

**Day 2 – Rome**
- Morning: Vatican Museums (8:30–11:30 AM, ~$25)
- Afternoon: Explore Piazza Navona (~Free)
- Evening: Gelato by the Pantheon (~$5)
---

Tone:
- Friendly and informative.
- Be concise but specific.
- Ensure the plan feels achievable and human-centered.

Objective:
Deliver a well-paced, realistic, and inspiring itinerary that reflects the user’s intent.
"""

reviewer_agent = Agent(
    name="Reviewer Agent",
    model="openai.gpt-4o",
    instructions=REVIEWER_INSTRUCTIONS.strip(),
    tools=[internet_search]
)

planner_agent = Agent(
    name="Planner Agent",
    model="openai.gpt-4o",
    instructions=PLANNER_INSTRUCTIONS.strip(),
)

# END SOLUTION


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration Helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_text(result_obj: Any) -> str:
    """
    Pull a usable string from the Runner result in a tolerant way.
    Your Runner may expose final_output, text, or __str__.
    """
    return (
        getattr(result_obj, "final_output", None)
        or getattr(result_obj, "text", None)
        or str(result_obj)
    )


def run_planner(user_text: str) -> str:
    """Run the Planner and return its itinerary text."""
    result = asyncio.run(Runner.run(planner_agent, user_text))
    return extract_text(result)


def run_reviewer(plan_text: str) -> str:
    """Run the Reviewer on the planner’s output and return validated text."""
    result = asyncio.run(Runner.run(reviewer_agent, plan_text))
    return extract_text(result)


# ──────────────────────────────────────────────────────────────────────────────
# Mock Data for Testing
# ──────────────────────────────────────────────────────────────────────────────

MOCK_RESULTS = {
    "planner_draft": """
**Day 1 – Arrival in Paris**
- Morning: Check-in at hostel (~$50)
- Afternoon: Visit Eiffel Tower (~$20)
- Evening: Dinner (~$30)

**Day 2 – Louvre**
- Morning: Louvre Museum (~$25)
- Afternoon: Walk along Seine (~$0)
- Evening: Bistro dinner (~$40)
""",
    "review_summary": """
### Review Summary
Overall good, but some prices outdated.

### Delta List (Suggested Fixes)
1. Eiffel Tower price — Outdated — Update to $30
2. Louvre — Correct
""",
    "delta_list": ["Eiffel Tower price outdated", "Louvre correct"],
    "final_itinerary": """
**Day 1 – Arrival in Paris**
- Morning: Check-in at hostel (~$50)
- Afternoon: Visit Eiffel Tower (~$30)
- Evening: Dinner (~$30)

**Day 2 – Louvre**
- Morning: Louvre Museum (~$25)
- Afternoon: Walk along Seine (~$0)
- Evening: Bistro dinner (~$40)
""",
    "tool_log": ["Tool call: internet_search for Eiffel Tower price", "Result: $30"]
}


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Travel Planner", page_icon="✈️", layout="wide")

# Sidebar
with st.sidebar:
    st.title("✈️ Multi-Agent Travel Planner")
    st.markdown("""
    This app uses two AI agents to plan your trip:
    - **Planner Agent**: Creates a detailed itinerary.
    - **Reviewer Agent**: Validates with real-time fact-checking.
    """)
    
    st.subheader("Agent Flow")
    st.graphviz_chart(render_flow_diagram())
    
    st.subheader("Tips")
    st.info("💡 Provide details like destination, duration, budget, and interests for better results.")
    st.warning("⚠️ The Planner relies on general knowledge; Reviewer checks facts online.")

# Main area
st.title("🧭 Multi-Agent Travel Planner")
st.markdown("Enter your travel prompt below and let the agents create and validate your itinerary.")

mock_mode = st.checkbox("Use Mock Data for Testing")

# Input
user_prompt = st.text_area("Travel Prompt", placeholder="E.g., Plan a 5-day trip to Paris for two people with a $2000 budget, focusing on art and food.", height=100)

# Button
if st.button("Run Planner → Reviewer", type="primary", use_container_width=True):
    if not user_prompt.strip() and not mock_mode:
        st.error("Please enter a travel prompt or use mock data.")
    else:
        if mock_mode:
            st.success("Using mock data for testing.")
            st.session_state.results = MOCK_RESULTS
        else:
            TOOL_LOGS.clear()
            with st.status("Processing...", expanded=True) as status:
                st.write("🧭 Running Planner Agent...")
                plan_text = run_planner(user_prompt)
                st.write("🔍 Running Reviewer Agent...")
                review_text = run_reviewer(plan_text)
                status.update(label="✅ Complete!", state="complete")
            
            # Store results
            st.session_state.results = {
                "planner_draft": plan_text,
                "review_summary": review_text,
                "final_itinerary": review_text,  # For now, same as review
                "tool_log": TOOL_LOGS.copy()
            }

# Display tabs if results exist
if "results" in st.session_state:
    results = st.session_state.results
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Planner Draft", "Reviewer Check", "Final Itinerary", "Cost Visualization", "Logs"])
    
    with tab1:
        st.header("🧭 Planner Draft")
        days = split_days(results["planner_draft"])
        for day in days:
            with st.expander(day["title"]):
                st.markdown(day["content"])
    
    with tab2:
        st.header("🔍 Reviewer Check")
        st.markdown(results["review_summary"])
    
    with tab3:
        st.header("✅ Final Itinerary")
        days = split_days(results["final_itinerary"])
        for day in days:
            with st.expander(day["title"]):
                st.markdown(day["content"])
    
    with tab4:
        st.header("💰 Cost Visualization")
        costs = parse_currency(results["final_itinerary"])
        if costs:
            chart = render_cost_chart(costs)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No costs found in the itinerary.")
    
    with tab5:
        st.header("📋 Logs")
        if results["tool_log"]:
            for log in results["tool_log"]:
                st.code(log)
        else:
            st.info("No tool logs available.")

