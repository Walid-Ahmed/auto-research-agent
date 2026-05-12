# Research Report Agent with Planning

A multi-agent system that autonomously plans, researches, writes, and edits a research report on any topic — powered by **OpenAI or Anthropic models** (switch providers by changing model names in `config.py`).

---

## File Structure

```
research_report_w_planning_agent/
│
├── main.py          # Agent definitions and entry point
├── config.py        # API keys, model names, and topic config
├── prompts.py       # All prompt templates for each agent
├── tools.py         # Tool definitions and dispatch handler
├── w_utils.py       # Shared utilities: call_model, run_agent_loop, usage tracking
│
├── plan.txt         # Generated at runtime — the planner's step list
├── report.md        # Generated at runtime — the final written report
├── usage_summary.txt  # Generated at runtime — token counts and cost breakdown
│
├── .env             # API keys (not committed)
└── .gitignore
```

---

## File Roles

| File | Role |
|---|---|
| `main.py` | Defines all agents and orchestrates the workflow |
| `config.py` | Stores model names, topic, and API key loading |
| `prompts.py` | Prompt templates for planner, researcher, writer, editor, and executor |
| `tools.py` | Tool implementations (Tavily, Wikipedia, arXiv) and the unified `handle_tool_call` dispatcher |
| `w_utils.py` | `call_model` (single-shot), `run_agent_loop` (tool-use loop), `get_usage`, `summarize_usages`, `clean_json_block` |

---

## Provider Support

The system is **provider-agnostic**. `w_utils.py` routes every call automatically based on the model name:

- Any model name containing `"claude"` or `"anthropic"` → **Anthropic SDK** (`anthropic.Anthropic`)
- Everything else → **OpenAI SDK** (`openai.OpenAI`)

No agent code changes are needed when switching providers. Just update the model names in `config.py`.

### Switching between OpenAI and Anthropic

```python
# config.py — use Anthropic Claude
planner_model  = "claude-sonnet-4-6"
writer_model   = "claude-sonnet-4-6"
executor_model = "claude-sonnet-4-6"
editor_model   = "claude-sonnet-4-6"

# config.py — use OpenAI
planner_model  = "gpt-4o"
writer_model   = "gpt-4o"
executor_model = "gpt-4o"
editor_model   = "gpt-4o"
```

### Supported models

| Provider | Model ID | Context | Input $/1M | Output $/1M |
|---|---|---|---|---|
| Anthropic | `claude-opus-4-7` | 1M | $5.00 | $25.00 |
| Anthropic | `claude-opus-4-6` | 1M | $5.00 | $25.00 |
| Anthropic | `claude-sonnet-4-6` | 1M | $3.00 | $15.00 |
| Anthropic | `claude-haiku-4-5` | 200K | $1.00 | $5.00 |
| OpenAI | `gpt-4o` | 128K | $2.50 | $10.00 |
| OpenAI | `o4-mini` | 200K | $1.10 | $4.40 |
| OpenAI | `o3` | 200K | $2.00 | $8.00 |

---

## Agents

| Agent | Role |
|---|---|
| **Planner** | Generates a step-by-step research plan as a Python list |
| **Executor** | Reads each step, decides which agent to call, passes enriched context |
| **Research** | Searches Wikipedia, arXiv, and the web using a tool-use loop |
| **Writer** | Drafts the final report from accumulated research context |
| **Editor** | Reflects on the draft and produces a revised, polished version |

---

## Workflow

```mermaid
flowchart TD
    A([Start]) --> B[Planner Agent\nGenerates step-by-step plan]
    B --> C[Save plan.txt]
    C --> D[Executor Agent\nLoops over each step]

    D --> E{Decide agent\nfor this step}

    E -->|research_agent| F[Research Agent\nSearches Wikipedia · arXiv · Web]
    F --> F1[Tool: wikipedia_search_tool]
    F --> F2[Tool: tavily_search_tool]
    F --> F3[Tool: arxiv_search_tool]
    F1 & F2 & F3 --> G[Return findings]

    E -->|writer_agent| H[Writer Agent\nDrafts report from context]
    H --> I[Save report.md]

    E -->|editor_agent| J[Editor Agent\nRevises and polishes draft]

    G & I & J --> K[Append to history\nBuild context for next step]
    K --> D

    D --> L[Print usage summary]
    L --> M([Done])
```

---

## Setup

1. Clone the repo and install dependencies:
```bash
pip install openai anthropic tavily-python wikipedia python-dotenv pillow
```

2. Create a `.env` file:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

3. Set your topic and models in `config.py`:
```python
topic = "Java Programming Language"

planner_model  = "claude-sonnet-4-6"   # or any OpenAI model
writer_model   = "claude-sonnet-4-6"
executor_model = "claude-sonnet-4-6"
editor_model   = "claude-sonnet-4-6"
```

4. Run:
```bash
python main.py
```

---

## Output

| File | Description |
|---|---|
| `plan.txt` | Numbered list of steps the planner generated |
| `report.md` | Final research report in Markdown format |
| `usage_summary.txt` | Token counts and cost breakdown for every agent call |
| Console | Per-step agent output + usage/cost summary table |

---

## Sample Plan (`plan.txt`)

The planner autonomously decides how many steps to generate and which agents to use. Here is a typical output for the topic **"Java Programming Language"**:

```
1. Use the research agent to search Wikipedia for the history and overview of the Java programming language
2. Use the research agent to search arXiv for academic papers on Java performance, concurrency, and JVM internals
3. Use the research agent to search the web for recent Java ecosystem trends, tooling, and community updates
4. Use the writer agent to draft a comprehensive research report based on all gathered findings
5. Use the editor agent to review, revise, and polish the draft into a final publication-ready report
```

And for **"Quantum Computing"**:

```
1. Use the research agent to search Wikipedia for an overview of quantum computing principles
2. Use the research agent to search arXiv for recent breakthroughs in quantum algorithms and hardware
3. Use the writer agent to synthesize findings into a structured technical report
4. Use the editor agent to refine the report for clarity and accuracy
```

---

## Usage Summary Example

The following is the actual output from a 12-step run on **"Java Programming Language"** using `claude-sonnet-4-6`:

```
======================================================================
📊 USAGE SUMMARY
======================================================================
  Step 1 executor                          122 in       164 out  $0.00086
  Step 1 research_agent                  1,766 in     2,016 out  $0.01081
  Step 2 executor                          121 in       185 out  $0.00095
  Step 2 research_agent                  8,076 in     1,445 out  $0.01524
  Step 3 executor                          124 in       140 out  $0.00075
  Step 3 research_agent                 19,516 in     2,633 out  $0.03305
  Step 4 executor                          122 in       306 out  $0.00148
  Step 4 research_agent                  2,979 in     1,515 out  $0.00994
  Step 5 executor                          124 in       170 out  $0.00088
  Step 5 writer_agent                    4,002 in     1,323 out  $0.01022
  Step 6 executor                          115 in       132 out  $0.00071
  Step 6 editor_agent                    4,864 in     1,133 out  $0.01034
  Step 7 executor                          114 in       220 out  $0.00109
  Step 7 writer_agent                    5,882 in     2,000 out  $0.01527
  Step 8 executor                          131 in       235 out  $0.00118
  Step 8 writer_agent                    6,430 in     1,813 out  $0.01505
  Step 9 executor                          119 in       263 out  $0.00129
  Step 9 editor_agent                    7,618 in     1,293 out  $0.01407
  Step 10 executor                         119 in       189 out  $0.00096
  Step 10 writer_agent                   8,492 in     2,000 out  $0.01814
  Step 11 executor                         123 in       218 out  $0.00109
  Step 11 editor_agent                   9,748 in     1,909 out  $0.01912
  Step 12 executor                         120 in       155 out  $0.00081
  Step 12 writer_agent                  11,030 in     2,000 out  $0.02093
----------------------------------------------------------------------
  TOTAL                                 91,857 in    23,457 out  $0.20425
======================================================================
```
