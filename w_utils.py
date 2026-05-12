# =============================================================================
# w_utils.py — Shared utilities for the multi-agent research pipeline
#
# Provides a single call_model() entry point that routes to either the
# Anthropic or OpenAI SDK based on the model name. All agents in main.py
# call only call_model() or run_agent_loop() — no provider-specific code
# leaks into the agent layer.
# =============================================================================

from dotenv import load_dotenv
from typing import Any
import os
import re
import openai
import anthropic

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialise both clients up front. Whichever isn't used costs nothing.
openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else openai.OpenAI()
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else anthropic.Anthropic()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def call_model(
    model: str,
    prompt: str | list,
    image: tuple[str, str] | None = None,
    max_tokens: int = 2000,
    thinking_budget: int | None = None,
    temperature: float = 0.2,
    tools: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Unified single-shot model call. Routes to Claude or OpenAI by model name.

    Args:
        model:           Model name, e.g. "claude-sonnet-4-6", "gpt-4o", "o4-mini".
                         Routing is automatic: any name containing "claude" or
                         "anthropic" goes to the Anthropic SDK; everything else
                         goes to the OpenAI SDK.
        prompt:          Either a plain string (wrapped into a user message
                         automatically) or a list of message dicts with "role"
                         and "content" keys (e.g. system + user turns).
        image:           Optional (media_type, base64_string) tuple for multimodal
                         input. Prepended to the user content.
        max_tokens:      Maximum output tokens.
        thinking_budget: Enables extended reasoning. For Anthropic this is a token
                         budget; for OpenAI it maps to "low"/"medium"/"high" effort.
        temperature:     Sampling temperature. Skipped automatically for models
                         that don't support it (Opus 4.7, OpenAI reasoning models).
        tools:           Tool definitions in OpenAI function-call format. Converted
                         to Anthropic format automatically when routing to Claude.

    Returns a dict with keys:
        "text"     — final visible text response
        "thinking" — extended reasoning / thinking content (if any)
        "tools"    — list of tool calls the model made: [{id, name, input}]
        "usage"    — {"input_tokens": int, "output_tokens": int}
        "raw"      — the original SDK response object
    """
    is_anthropic = "claude" in model.lower() or "anthropic" in model.lower()

    if is_anthropic:
        return _call_anthropic(model, prompt, image, max_tokens, thinking_budget, temperature, tools)
    return _call_openai(model, prompt, image, max_tokens, thinking_budget, temperature, tools)


def run_agent_loop(model, prompt, tools, dispatch_fn, max_iterations=10, temperature=0.2):
    """
    Agentic tool-use loop: keeps calling the model and executing tool calls
    until the model stops requesting tools or max_iterations is reached.

    Args:
        model:         Model name (routing is automatic, same as call_model).
        prompt:        Initial user prompt string.
        tools:         Tool definitions in OpenAI function-call format.
        dispatch_fn:   Callable(name, args) that executes a tool and returns
                       a JSON-serialisable result.
        max_iterations: Safety cap on the number of model calls.
        temperature:   Sampling temperature (skipped for models that don't support it).

    Returns:
        (text: str, usage: dict)  — final text response and accumulated token counts.
    """
    is_anthropic = "claude" in model.lower() or "anthropic" in model.lower()
    if is_anthropic:
        return _anthropic_loop(model, prompt, tools, dispatch_fn, max_iterations, temperature)
    return _openai_loop(model, prompt, tools, dispatch_fn, max_iterations, temperature)


# -----------------------------------------------------------------------------
# Tool normalisation
# -----------------------------------------------------------------------------

def _normalize_tools_for_anthropic(tools: list[dict]) -> list[dict]:
    """
    Convert OpenAI-style tool defs  {"type": "function", "function": {...}}
    to Anthropic format             {"name": ..., "description": ..., "input_schema": {...}}.

    Tools already in Anthropic format are passed through unchanged, so this
    function is safe to call unconditionally.
    """
    normalized = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            fn = t["function"]
            normalized.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                # Anthropic uses "input_schema" where OpenAI uses "parameters"
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        else:
            normalized.append(t)
    return normalized


# -----------------------------------------------------------------------------
# Anthropic backend
# -----------------------------------------------------------------------------

def _call_anthropic(model, prompt, image, max_tokens, thinking_budget, temperature: float = 0.2, tools=None):
    """Single-shot call to the Anthropic Messages API."""

    # Build the messages list from either a plain string or a list of message dicts.
    if isinstance(prompt, list):
        # Extract the system message (Anthropic takes it as a top-level param,
        # not as a message role) and collect the remaining turns.
        system_content = None
        messages = []
        for msg in prompt:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                messages.append(msg)

        # Attach image to the last user message when both a list and an image are provided.
        if image and messages:
            media_type, b64 = image
            last = messages[-1]
            content = last.get("content", [])
            if isinstance(content, str):
                # Convert plain-string content to a content block list
                content = [{"type": "text", "text": content}]
            # Image must come before text in Anthropic's content array
            content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}})
            messages[-1] = {**last, "content": content}
    else:
        # Plain string prompt — wrap into a single user message.
        system_content = None
        content = []
        if image:
            media_type, b64 = image
            # Image block must precede the text block
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64},
            })
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    if system_content:
        kwargs["system"] = system_content

    # Opus 4.7 removed temperature support entirely — sending it returns a 400.
    if "opus-4-7" not in model.lower():
        kwargs["temperature"] = temperature

    if thinking_budget:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

    if tools:
        # Tools arrive in OpenAI format from the caller; convert before sending.
        kwargs["tools"] = _normalize_tools_for_anthropic(tools)

    msg = anthropic_client.messages.create(**kwargs)

    # Parse the response content blocks into a flat structure.
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


def _anthropic_loop(model, prompt, tools, dispatch_fn, max_iterations, temperature):
    """
    Agentic tool-use loop for Anthropic models.

    Maintains a growing messages list, appending assistant turns (including
    tool_use blocks) and tool_result turns until the model stops requesting tools.
    """
    import json

    messages = [{"role": "user", "content": prompt}]
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    for _ in range(max_iterations):
        msg = anthropic_client.messages.create(
            model=model,
            messages=messages,
            tools=_normalize_tools_for_anthropic(tools),
            max_tokens=2000,
            temperature=temperature,
        )
        total_usage["input_tokens"] += msg.usage.input_tokens
        total_usage["output_tokens"] += msg.usage.output_tokens

        # Append the full assistant response (including tool_use blocks) so the
        # next turn has the complete conversation context.
        messages.append({"role": "assistant", "content": msg.content})

        tool_calls = [b for b in msg.content if b.type == "tool_use"]
        if not tool_calls:
            # No more tool calls — return the final text response.
            return "".join(b.text for b in msg.content if b.type == "text"), total_usage

        # Execute each tool and collect results to send back in one user turn.
        tool_results = []
        for tc in tool_calls:
            result = dispatch_fn(tc.name, tc.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})

    return "", total_usage


# -----------------------------------------------------------------------------
# OpenAI backend
# -----------------------------------------------------------------------------

def _call_openai(model, prompt, image, max_tokens, thinking_budget, temperature: float = 0.2, tools=None):
    """Single-shot call to the OpenAI Responses API."""

    # o3/o4/gpt-5 reasoning models do not accept a temperature parameter.
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

    if not is_reasoning_model:
        kwargs["temperature"] = temperature

    if thinking_budget:
        # Map a token budget to OpenAI's coarse effort levels.
        effort = "low" if thinking_budget < 4000 else "medium" if thinking_budget < 16000 else "high"
        kwargs["reasoning"] = {"effort": effort, "summary": "auto"}

    if tools:
        # Flatten OpenAI's {"type": "function", "function": {...}} wrapper into
        # the flat format that the Responses API expects.
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

    # Extract reasoning summaries and function calls from the output items.
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
        "text": response.output_text or "",  # convenience property that joins all text items
        "thinking": "".join(thinking_parts),
        "tools": tool_calls,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "raw": response,
    }


def _openai_loop(model, prompt, tools, dispatch_fn, max_iterations, temperature):
    """
    Agentic tool-use loop for OpenAI models using the Responses API.

    Uses previous_response_id to chain turns efficiently — the server keeps
    the conversation state so we only need to send tool outputs each round.
    """
    import json

    # Flatten tool definitions for the Responses API
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

        # Execute all tool calls and bundle results for the next turn.
        tool_outputs = []
        for tc in tool_calls:
            args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
            result = dispatch_fn(tc.name, args)
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": tc.call_id,
                "output": json.dumps(result),
            })

        # Continue the conversation by referencing the previous response ID.
        cont_kwargs = {"model": model, "previous_response_id": response.id, "input": tool_outputs, "tools": normalized}
        if not is_reasoning:
            cont_kwargs["temperature"] = temperature
        response = openai_client.responses.create(**cont_kwargs)
        total_usage["input_tokens"] += response.usage.input_tokens
        total_usage["output_tokens"] += response.usage.output_tokens

    return response.output_text or "", total_usage


# -----------------------------------------------------------------------------
# Usage & cost tracking
# -----------------------------------------------------------------------------

def calculate_cost(usage: dict, model: str = "o4-mini") -> float:
    """
    Estimate cost in USD from a usage dict.
    Rates are per 1M tokens (input / output). Unknown models fall back to o4-mini rates.
    """
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    # Pricing as of May 2026 ($/1M tokens)
    pricing = {
        # OpenAI
        "o4-mini":      {"input": 1.10,  "output": 4.40},
        "o3":           {"input": 2.00,  "output": 8.00},
        "gpt-5.4-mini": {"input": 0.75,  "output": 4.50},
        "gpt-5.4":      {"input": 2.50,  "output": 15.00},
        # Anthropic
        "claude-opus-4-7":   {"input": 5.00,  "output": 25.00},
        "claude-opus-4-6":   {"input": 5.00,  "output": 25.00},
        "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
        "claude-haiku-4-5":  {"input": 1.00,  "output": 5.00},
    }

    model_key = model.lower().replace("openai:", "").replace("anthropic:", "")
    rates = pricing.get(model_key, {"input": 1.10, "output": 4.40})

    cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
    return round(cost, 6)


def get_usage(response: dict, model: str, task_name: str | None = None) -> str:
    """Format a single usage dict into a human-readable report string."""
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


def summarize_usages(all_usages: list[tuple[str, str, dict]]) -> str:
    """
    Build a cost summary table from a list of (label, model, usage_dict) tuples.
    Printed at the end of a full pipeline run.
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

from pathlib import Path
import base64
import mimetypes


def encode_image_b64(image_path: str) -> tuple[str, str]:
    """Read an image file and return (media_type, base64_string)."""
    path = Path(image_path)
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type is None:
        media_type = "image/png"  # matplotlib default

    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.standard_b64encode(raw).decode("utf-8")
    return media_type, b64


def clean_json_block(raw: str) -> str:
    """
    Strip Markdown code fences from a model response so the inner content
    can be parsed as JSON or a Python literal.
    Handles any language tag: ```json, ```python, ``` etc.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


if __name__ == "__main__":
    pass
