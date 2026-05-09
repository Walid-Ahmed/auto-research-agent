from dotenv import load_dotenv
from typing import Any
import os
import openai
import anthropic

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
# Both clients read keys from env by default; explicit is also fine:
openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else openai.OpenAI()
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else anthropic.Anthropic()



def call_model(
    model: str,
    prompt: str | list,
    image: tuple[str, str] | None = None,
    max_tokens: int = 2000,
    thinking_budget: int | None = None,
    temperature: float = 0.2,        # ← Good default for agents
    tools: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Unified call. Routes to Claude or OpenAI by model name.

    Args:
        model: Model name (e.g. "claude-opus-4-7", "gpt-5", "o3").
        prompt: User prompt text. or list
        image: Optional (media_type, b64) tuple for multimodal input.
        max_tokens: Max output tokens.
        thinking_budget: If set, enables extended thinking (Anthropic) or
            reasoning effort (OpenAI). For Anthropic this is the literal token
            budget; for OpenAI it's mapped to "low"/"medium"/"high".
        tools: Optional list of tool definitions in the provider's native format.

    Returns:
        {
            "text":     str,                  # final visible text
            "thinking": str,                  # reasoning/thinking content
            "tools":    list[dict],           # tool calls with name, input, id
            "usage":    dict,                 # token counts
            "raw":      Any,                  # original SDK response object
        }
        :param temperature:
    """
    # Route by model name prefix — avoids needing separate config flags per provider
    is_anthropic = "claude" in model.lower() or "anthropic" in model.lower()


    if is_anthropic:
        return _call_anthropic(
            model,
            prompt,
            image,
            max_tokens,
            thinking_budget,
            temperature,      # ← Pass temperature
            tools
        )
    return _call_openai(model, prompt, image, max_tokens, thinking_budget, temperature,tools)


def _call_anthropic(model, prompt, image, max_tokens, thinking_budget, temperature: float = 0.2 ,tools=None):
    content = []
    if image:
        media_type, b64 = image
        # Image must come before text in the content array for Anthropic's API
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        })
    content.append({"type": "text", "text": prompt})

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,  # ← Added this line
        "messages": [{"role": "user", "content": content}],
    }
    if thinking_budget:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    if tools:
        kwargs["tools"] = tools

    msg = anthropic_client.messages.create(**kwargs)

    text_parts, thinking_parts, tool_calls = [], [], []
    for block in msg.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "thinking":
            thinking_parts.append(block.thinking)
        elif block.type == "tool_use":
            tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

    return {
        "text": "".join(text_parts),
        "thinking": "".join(thinking_parts),
        "tools": tool_calls,
        "usage": {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        },
        "raw": msg,
    }


def _call_openai(model, prompt, image, max_tokens, thinking_budget,temperature: float = 0.2, tools=None):
    # Detect reasoning models (they don't support temperature)
    is_reasoning_model = any(x in model.lower() for x in ["o3", "o4", "gpt-5", "gpt5"])


    if image:
        media_type, b64 = image
        input_payload = [{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"data:{media_type};base64,{b64}"},
                {"type": "input_text", "text": prompt},
            ],
        }]
    else:
        input_payload = prompt

    kwargs = {
        "model": model,
        "input": input_payload,
        "max_output_tokens": max_tokens,
    }

    # Only add temperature for NON-reasoning models
    if not is_reasoning_model:
        kwargs["temperature"] = temperature

    if thinking_budget:
        # Map a token budget to OpenAI's effort levels (rough heuristic).
        effort = "low" if thinking_budget < 4000 else "medium" if thinking_budget < 16000 else "high"
        kwargs["reasoning"] = {"effort": effort, "summary": "auto"}
    if tools:
        normalized = []
        for t in tools:
            if t.get("type") == "function" and "function" in t:
                fn = t["function"]
                normalized.append({
                    "type": "function",
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
            else:
                normalized.append(t)
        kwargs["tools"] = normalized

    response = openai_client.responses.create(**kwargs)

    thinking_parts, tool_calls = [], []
    for item in getattr(response, "output", []) or []:
        itype = getattr(item, "type", None)
        if itype == "reasoning":
            for part in getattr(item, "summary", []) or []:
                t = getattr(part, "text", "")
                if t:
                    thinking_parts.append(t)
        elif itype == "function_call":
            tool_calls.append({
                "id": getattr(item, "call_id", None),
                "name": getattr(item, "name", None),
                "input": getattr(item, "arguments", None),
            })

    return {
        # output_text is a convenience property that joins all text output items
        "text": response.output_text or "",
        "thinking": "".join(thinking_parts),
        "tools": tool_calls,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "raw": response,
    }








from pathlib import Path
import base64
import mimetypes
from pathlib import Path


def encode_image_b64(image_path: str) -> tuple[str, str]:
    """
    Read an image file and return (media_type, base64_string).
    media_type is something like 'image/png' or 'image/jpeg'.
    """
    path = Path(image_path)

    # Guess media type from the file extension
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type is None:
        # Fallback: assume PNG since matplotlib defaults to PNG
        media_type = "image/png"

    # Read raw bytes and encode
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.standard_b64encode(raw).decode("utf-8")

    return media_type, b64


def calculate_cost(usage: dict, model: str = "o4-mini") -> float:
    """
    Convert usage dict to estimated cost in USD.
    Returns cost as float (e.g. 0.0123 = $0.0123)
    """
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    # Current pricing (May 2026)
    pricing = {
        "o4-mini": {"input": 1.10, "output": 4.40},
        "o3": {"input": 2.00, "output": 8.00},
        "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
        "gpt-5.4": {"input": 2.50, "output": 15.00},
        # Add more models as needed
    }

    model_key = model.lower().replace("openai:", "")
    rates = pricing.get(model_key, {"input": 1.10, "output": 4.40})  # default to o4-mini

    cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
    return round(cost, 6)


def get_usage(response: dict, model: str, task_name: str | None = None) -> str:
    cost = calculate_cost(response["usage"], model=model)
    usage = response.get("usage", {})

    input_t = usage.get("input_tokens", 0)
    output_t = usage.get("output_tokens", 0)
    total_t = input_t + output_t

    header = f"📊 USAGE REPORT → {task_name}" if task_name else "📊 USAGE REPORT"
    lines = [
        "=" * 70,
        header,
        "=" * 70,
        f"Input tokens   : {input_t:,}",
        f"Output tokens  : {output_t:,}",
        f"Total tokens   : {total_t:,}",
        f"Estimated cost : ${cost:.5f}",
        "=" * 70,
    ]
    return "\n".join(lines)

def run_agent_loop(model, prompt, tools, dispatch_fn, max_iterations=10, temperature=0.2):
    """Run an agentic tool-use loop until the model stops requesting tools."""
    is_anthropic = "claude" in model.lower() or "anthropic" in model.lower()
    if is_anthropic:
        return _anthropic_loop(model, prompt, tools, dispatch_fn, max_iterations, temperature)
    return _openai_loop(model, prompt, tools, dispatch_fn, max_iterations, temperature)


def _openai_loop(model, prompt, tools, dispatch_fn, max_iterations, temperature):
    import json

    normalized = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            fn = t["function"]
            normalized.append({
                "type": "function",
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        else:
            normalized.append(t)

    is_reasoning = any(x in model.lower() for x in ["o3", "o4", "gpt-5", "gpt5"])
    kwargs = {"model": model, "input": prompt, "tools": normalized}
    if not is_reasoning:
        kwargs["temperature"] = temperature

    response = openai_client.responses.create(**kwargs)
    total_usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}

    for _ in range(max_iterations):
        tool_calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
        if not tool_calls:
            return response.output_text or "", total_usage

        tool_outputs = []
        for tc in tool_calls:
            args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
            result = dispatch_fn(tc.name, args)
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": tc.call_id,
                "output": json.dumps(result),
            })

        cont_kwargs = {"model": model, "previous_response_id": response.id, "input": tool_outputs, "tools": normalized}
        if not is_reasoning:
            cont_kwargs["temperature"] = temperature
        response = openai_client.responses.create(**cont_kwargs)
        total_usage["input_tokens"] += response.usage.input_tokens
        total_usage["output_tokens"] += response.usage.output_tokens

    return response.output_text or "", total_usage


def _anthropic_loop(model, prompt, tools, dispatch_fn, max_iterations, temperature):
    import json

    messages = [{"role": "user", "content": prompt}]
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    for _ in range(max_iterations):
        msg = anthropic_client.messages.create(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=2000,
            temperature=temperature,
        )
        total_usage["input_tokens"] += msg.usage.input_tokens
        total_usage["output_tokens"] += msg.usage.output_tokens
        messages.append({"role": "assistant", "content": msg.content})

        tool_calls = [b for b in msg.content if b.type == "tool_use"]
        if not tool_calls:
            return "".join(b.text for b in msg.content if b.type == "text"), total_usage

        tool_results = []
        for tc in tool_calls:
            result = dispatch_fn(tc.name, tc.input)
            tool_results.append({"type": "tool_result", "tool_use_id": tc.id, "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})

    return "", total_usage

def summarize_usages(all_usages: list[tuple[str, str, dict]]) -> str:
    """
    Print a summary table of all usage entries.
    Each entry is (label, model, usage_dict).
    """
    total_input, total_output, total_cost = 0, 0, 0.0
    rows = []
    for label, model, usage in all_usages:
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        cost = calculate_cost(usage, model=model)
        total_input += inp
        total_output += out
        total_cost += cost
        rows.append(f"  {label:<35} {inp:>8,} in  {out:>8,} out  ${cost:.5f}")

    lines = [
        "=" * 70,
        "📊 USAGE SUMMARY",
        "=" * 70,
    ] + rows + [
        "-" * 70,
        f"  {'TOTAL':<35} {total_input:>8,} in  {total_output:>8,} out  ${total_cost:.5f}",
        "=" * 70,
    ]
    return "\n".join(lines)


def clean_json_block(raw: str) -> str:
    """
    Clean the contents of a JSON block that may come wrapped with Markdown backticks.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()

if __name__ == "__main__":
  pass