#!/usr/bin/env python3
"""Reproduce automatic tool-call formatting against a running ninfer-serve."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from typing import Any


TOOL = {
    "type": "function",
    "function": {
        "name": "Bash",
        "description": "Run one shell command when the user explicitly asks.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command."},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


def post(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            value = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    if not isinstance(value, dict):
        raise RuntimeError("server returned a non-object JSON response")
    return value


def summarize(response: dict[str, Any]) -> dict[str, Any]:
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    parsed_arguments: list[Any] = []
    argument_errors: list[str] = []
    if isinstance(tool_calls, list):
        for call in tool_calls:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            raw = function.get("arguments")
            try:
                parsed_arguments.append(json.loads(raw))
            except (TypeError, json.JSONDecodeError) as error:
                argument_errors.append(str(error))
    markers = None
    function_name = None
    if isinstance(content, str):
        markers = {
            "tool_call": "<tool_call>" in content,
            "parameter_open": bool(re.search(r"<parameter=[^>]+>", content)),
            "parameter_close": "</parameter>" in content,
        }
        match = re.search(r"<function=([^>]+)>", content)
        function_name = match.group(1) if match else None
    return {
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "tool_calls": tool_calls,
        "parsed_arguments": parsed_arguments,
        "argument_errors": argument_errors,
        "content_markers": markers,
        "content_function_name": function_name,
        "usage": response.get("usage"),
    }


def payload(model: str, user_text: str, *, tools: bool, tool_choice: Any = None,
            temperature: float = 0, seed: int = 7) -> dict[str, Any]:
    value: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
        "max_tokens": 128,
        "temperature": temperature,
        "top_p": 1,
        "seed": seed,
        "enable_thinking": False,
    }
    if tools:
        value["tools"] = [TOOL]
        if tool_choice is not None:
            value["tool_choice"] = tool_choice
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="huihui-qwen3.8-27b-abliterated")
    args = parser.parse_args()

    cases = [
        ("plain", payload(args.model, "Reply with exactly OK. Do not call any tool.", tools=False)),
        ("tool_auto_short", payload(args.model, "Use Bash to run exactly: printf TOOL_OK", tools=True, tool_choice="auto")),
        (
            "tool_forced",
            payload(
                args.model,
                "Use Bash to run exactly: printf TOOL_OK",
                tools=True,
                tool_choice={"type": "function", "function": {"name": "Bash"}},
            ),
        ),
        (
            "tool_auto_search_prompt",
            payload(
                args.model,
                "Let me search for the character. Use Bash to search the repository for "
                "dance_undress and the phrase revealing the breast (露出).",
                tools=True,
                tool_choice="auto",
            ),
        ),
    ]
    for seed in (1, 2, 3):
        cases.append(
            (
                f"tool_auto_seed_{seed}",
                payload(
                    args.model,
                    "Use Bash to run exactly: printf TOOL_OK",
                    tools=True,
                    tool_choice="auto",
                    seed=seed,
                ),
            )
        )

    for name, request in cases:
        try:
            result = summarize(post(args.base_url, request))
        except (RuntimeError, urllib.error.URLError) as error:
            print(json.dumps({"case": name, "error": str(error)}, ensure_ascii=False))
            return 1
        print(json.dumps({"case": name, **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
