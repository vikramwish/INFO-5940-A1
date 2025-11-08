# Reflection on Multi-Agent Travel Planning App

### What Worked Well
The division of roles between the Planner and Reviewer agents was highly effective. The Planner’s prompt produced coherent, detailed itineraries that balanced cost, pacing, and interests without external data. Meanwhile, the Reviewer’s integration of the `internet_search` tool added a realistic layer of fact-checking — ensuring feasibility and improving trustworthiness. Streamlit’s tabbed interface and Graphviz visualization helped communicate the Planner → Reviewer → Final Output flow clearly, making the collaboration intuitive to users. The modular code structure (separate helper functions for parsing, visualization, and orchestration) also made debugging and testing straightforward.

### What Could Be Improved
One challenge was controlling verbosity and formatting consistency across agent outputs. Occasionally, the Planner produced overly detailed narratives, or the Reviewer’s “Delta List” repeated points unnecessarily. Prompt tuning helped, but future iterations could use few-shot examples or lightweight reinforcement (e.g., reward short, structured answers). The `internet_search` integration also had latency when multiple queries were fired, suggesting a need for better caching or batched lookups. Finally, cost extraction and visualization were heuristic-based — more robust parsing (e.g., regex tuning or named entity extraction) would improve accuracy.

### What I Learned
This project deepened my understanding of **multi-agent collaboration** and **prompt engineering**. Designing prompts that clearly define role boundaries, tone, and expected outputs proved to be as critical as writing the code itself. I learned how even small wording changes (“verify” vs. “fact-check”) can shift model behavior dramatically. I also gained hands-on experience integrating **LLM reasoning within an interactive UI**, balancing creativity (Planner) and validation (Reviewer). Most importantly, this assignment illustrated how thoughtful agent orchestration can transform vague human intent into actionable, validated outputs — a core concept in applied AI design.



