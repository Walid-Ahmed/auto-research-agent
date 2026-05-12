import config
import prompts
import w_utils
import ast
from datetime import datetime
import tools
import json

def planner_agent(max_steps=4 ,limit_steps: bool = True):
    prompt = prompts.planning_prompt.format(
            topic=config.topic
        )
    response=w_utils.call_model(model=config.planner_model,prompt=prompt, temperature=1)
    steps_str=response["text"]

    # ==================== USAGE INFO ====================
    usage=w_utils.get_usage(response, model=config.planner_model, task_name="planner_agent")

    plan_steps = ast.literal_eval(w_utils.clean_json_block(steps_str))

    if limit_steps:
        plan_steps = plan_steps[:min(len(plan_steps), max_steps)]
    return plan_steps,usage


def research_agent(task):
    research_tools = [tools.wikipedia_tool, tools.tavily_tool, tools.arxiv_tool]
    text, usage = w_utils.run_agent_loop(
        model=config.planner_model,
        prompt=task,
        tools=research_tools,
        dispatch_fn=tools.handle_tool_call,
        temperature=1,
    )
    return text, usage

def writer_agent(task):
    system_msg = {
        "role": "system",
        "content": prompts.writer_system_prompt
    }
    user_msg = {
        "role": "user",
        "content": task
    }
    messages = [system_msg, user_msg]
    response=w_utils.call_model(model=config.writer_model,prompt=messages, temperature=1)
    return response["text"], response["usage"]

'''
The Executor Agent
The executor_agent manages the workflow by executing each step of a given plan. It:

Decides which agent (research_agent, writer_agent, or editor_agent) should handle the step.
Builds context from the outputs of previous steps.
Sends the enriched task to the selected agent.
Collects and stores the results in a shared history.
Notice that planner_agent might return a long list of steps. Because of this, the maximum number of steps is set to a maximum of 4 to keep running time reasonable.
'''

def editor_agent(task):
    response = w_utils.call_model(
        model=config.editor_model,
        prompt=task,
        temperature=1,
    )
    return response["text"], response["usage"]

def executor_agent(plan_steps, model: str = "openai:gpt-4o"):
    history = []
    all_usages = []
    agent_registry = {
        "research_agent": research_agent,
        "editor_agent": editor_agent,
        "writer_agent": writer_agent,
    }

    for i, step in enumerate(plan_steps):
        response = w_utils.call_model(
            model=config.executor_model,
            prompt=[{"role": "user", "content": prompts.agent_decision_prompt.format(step=step)}],
            temperature=0,
        )
        all_usages.append((f"Step {i+1} executor", config.executor_model, response["usage"]))

        cleaned_json = w_utils.clean_json_block(response["text"])
        print(cleaned_json)
        agent_info = json.loads(cleaned_json)
        agent_name = agent_info["agent"]
        task = agent_info["task"]
        context = ""
        for j, (s, a, r) in enumerate(history):
            context += f"Step {j+1} executed by {a}:\n{r}\n"

        enriched_task = f"""
        You are {agent_name}.

        Here is the context of what has been done so far:
        {context}

        Your next task is:
        {task}
        """
        print(f"\n🛠️ Executing with agent: `{agent_name}` on task: {task}")

        if agent_name in agent_registry:
            output, agent_usage = agent_registry[agent_name](enriched_task)
            all_usages.append((f"Step {i+1} {agent_name}", config.executor_model, agent_usage))
            history.append((step, agent_name, output))
        else:
            output = f"⚠️ Unknown agent: {agent_name}"
            history.append((step, agent_name, output))

        print(f"✅ Output:\n{output}")

    summary = w_utils.summarize_usages(all_usages)
    print(summary)
    return history, summary



if __name__ == "__main__":
    plan_steps, usage = planner_agent(limit_steps=False)
    print(plan_steps)

    plan_file = "plan.txt"
    with open(plan_file, "w") as f:
        for i, step in enumerate(plan_steps, 1):
            f.write(f"{i}. {step}\n")
    print(f"Plan saved to {plan_file}")

    history, summary = executor_agent(plan_steps, model=config.planner_model)

    for step, agent_name, output in history:
        if agent_name == "writer_agent":
            with open("report.md", "w") as f:
                f.write(output)
            print("Report saved to report.md")
            break

    with open("usage_summary.txt", "w") as f:
        f.write(summary)
    print("Usage summary saved to usage_summary.txt")

    print("\n✓ Workflow completed successfully")
    print("=" * 34)



