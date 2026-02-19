#!/usr/bin/env python3
"""A minimal AI agent: REPL that talks to Claude with conversation memory and tools."""

import ast
import operator
import subprocess
import sys
from dotenv import load_dotenv
import anthropic

load_dotenv()

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Tool definitions (Anthropic tool_use format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression and return the numeric result. "
            "Supports standard arithmetic operators (+, -, *, /, **, //, %) and "
            "common math functions (abs, round, min, max, pow, sum). "
            "No imports or arbitrary code — safe expression evaluation only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate, e.g. '2 ** 10 + 42'.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a whitelisted shell command and return its stdout/stderr output. "
            "Only safe, read-only commands are permitted. "
            "Allowed commands: ls, pwd, echo, date, whoami, uname, cat, head, tail, wc, "
            "find, grep, df, du, env, printenv, hostname, uptime, ps, which."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute (must start with an allowed command).",
                }
            },
            "required": ["command"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool execution helpers
# ---------------------------------------------------------------------------

# Safe subset of builtins for calculator
_SAFE_NAMES = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sum": sum,
    "int": int,
    "float": float,
    "True": True,
    "False": False,
}

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """Recursively evaluate a parsed AST node using only safe operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]
        raise ValueError(f"Name not allowed: {node.id}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Operator not allowed: {op_type}")
        return _SAFE_OPERATORS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unary operator not allowed: {op_type}")
        return _SAFE_OPERATORS[op_type](_safe_eval(node.operand))
    elif isinstance(node, ast.Call):
        func = _safe_eval(node.func)
        if not callable(func):
            raise ValueError("Not callable")
        args = [_safe_eval(a) for a in node.args]
        return func(*args)
    elif isinstance(node, ast.List):
        return [_safe_eval(e) for e in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(_safe_eval(e) for e in node.elts)
    else:
        raise ValueError(f"Unsupported AST node: {type(node)}")


def run_calculator(expression: str) -> str:
    """Safely evaluate a math expression and return the result as a string."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Allowlist of safe commands (first token of the command must be in this set)
_ALLOWED_COMMANDS = {
    "ls", "pwd", "echo", "date", "whoami", "uname", "cat", "head", "tail",
    "wc", "find", "grep", "df", "du", "env", "printenv", "hostname",
    "uptime", "ps", "which",
}


def run_command(command: str) -> str:
    """Execute a whitelisted shell command and return combined stdout/stderr."""
    parts = command.strip().split()
    if not parts:
        return "Error: empty command"

    base_cmd = parts[0]
    # Strip path prefix (e.g. /bin/ls → ls) for allowlist check
    base_cmd_name = base_cmd.split("/")[-1]

    if base_cmd_name not in _ALLOWED_COMMANDS:
        return (
            f"Error: command '{base_cmd_name}' is not in the allowed list. "
            f"Allowed: {', '.join(sorted(_ALLOWED_COMMANDS))}"
        )

    try:
        result = subprocess.run(
            parts,
            shell=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        return output.strip() if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (10 s limit)"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

TOOL_HANDLERS = {
    "calculator": lambda inp: run_calculator(inp["expression"]),
    "run_command": lambda inp: run_command(inp["command"]),
}


def execute_tool(name: str, tool_input: dict) -> str:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Error: unknown tool '{name}'"
    return handler(tool_input)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent_turn(client: anthropic.Anthropic, messages: list) -> str:
    """
    Run one full agent turn (potentially multiple model calls if tool_use is
    involved) and return the final text response.
    """
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOLS,
            messages=messages,
        )

        # Collect any text blocks from this response
        text_parts = []
        tool_uses = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)

        # If the model is done (no tool calls), return accumulated text
        if response.stop_reason != "tool_use":
            return "".join(text_parts)

        # ---- Handle tool calls ----
        # Execute each tool and gather results
        tool_results = []
        for tu in tool_uses:
            print(f"  🔧 Tool call: {tu.name}({tu.input})")
            result_text = execute_tool(tu.name, tu.input)
            print(f"  ✅ Result: {result_text[:200]}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result_text,
                }
            )

        # Only extend history once both messages are ready — avoids partial
        # state corruption if an anthropic.APIError is raised on the next call.
        new_messages = [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results},
        ]
        messages.extend(new_messages)
        # Continue the loop — model will now produce its final response


def main():
    client = anthropic.Anthropic()
    messages = []

    print("🤖 Agent ready (tools: calculator, run_command). Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            assistant_text = run_agent_turn(client, messages)
            # Persist the final assistant text turn in history
            messages.append({"role": "assistant", "content": assistant_text})
            print(f"\nAgent: {assistant_text}\n")

        except anthropic.APIError as e:
            print(f"\n❌ API error: {e}\n")
            # Remove the failed user message so conversation stays valid
            messages.pop()


if __name__ == "__main__":
    main()
