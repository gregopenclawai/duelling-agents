#!/usr/bin/env python3
"""A minimal AI agent: REPL that talks to Claude with conversation memory and tools."""

import argparse
import ast
import json
import operator
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# SOUL.md — system prompt / personality loader
# ---------------------------------------------------------------------------

DEFAULT_SOUL_PATH = Path("SOUL.md")


def load_soul(soul_path: str | None = None) -> str | None:
    """
    Load a SOUL.md file and return its contents as a system prompt string.

    Resolution order:
    1. Explicit path passed via ``--soul`` CLI flag (``soul_path`` argument).
    2. ``SOUL.md`` in the current working directory.
    3. ``None`` — no system prompt (silent fallback).

    Returns the file contents as a string, or ``None`` if no file is found.
    """
    candidates: list[Path] = []
    if soul_path is not None:
        candidates.append(Path(soul_path))
    candidates.append(DEFAULT_SOUL_PATH)

    explicit = Path(soul_path) if soul_path is not None else None

    for path in candidates:
        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    print(f"🪬  Soul loaded from: {path}")
                    return content
            except OSError as e:
                print(f"⚠️  Could not read soul file '{path}': {e}", file=sys.stderr)
        elif explicit is not None and path == explicit:
            # Explicit path was given but file doesn't exist — warn loudly
            print(
                f"⚠️  Soul file '{soul_path}' not found — running without system prompt.",
                file=sys.stderr,
            )

    return None

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

    # Reject any command that contains a slash — only bare command names are
    # accepted. This prevents allowlist bypass via paths like ./ls or /bin/ls.
    if "/" in base_cmd:
        return (
            f"Error: command '{base_cmd}' must be a bare command name (no '/'). "
            f"Allowed: {', '.join(sorted(_ALLOWED_COMMANDS))}"
        )

    if base_cmd not in _ALLOWED_COMMANDS:
        return (
            f"Error: command '{base_cmd}' is not in the allowed list. "
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
# Session persistence (JSONL — one JSON object per line, append-only)
# ---------------------------------------------------------------------------

DEFAULT_SESSION_FILE = "./session.jsonl"


def _serialisable(obj):
    """Convert an Anthropic content block (or list thereof) to a JSON-safe value."""
    if isinstance(obj, list):
        return [_serialisable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    # Anthropic SDK model objects (e.g. TextBlock, ToolUseBlock) expose __dict__
    if hasattr(obj, "__dict__"):
        return {k: _serialisable(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    # str, int, float, bool, None are already JSON-safe
    return obj


def load_session(session_file: str) -> list:
    """
    Load messages from a JSONL session file.

    Returns an empty list if the file is missing.  Individual corrupted lines
    are skipped with a warning so a partially-written file is still recoverable.
    """
    path = Path(session_file)
    if not path.exists():
        return []

    messages = []
    skipped = 0
    try:
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    messages.append(json.loads(raw))
                except json.JSONDecodeError:
                    skipped += 1
                    print(f"  ⚠️  Skipping corrupted line {lineno} in {session_file}", file=sys.stderr)
    except OSError as exc:
        print(f"  ⚠️  Could not read session file '{session_file}': {exc}", file=sys.stderr)
        return []

    if skipped:
        print(f"  ⚠️  {skipped} corrupted line(s) ignored from {session_file}", file=sys.stderr)
    return messages


def append_message(session_file: str, message: dict) -> None:
    """
    Append a single message dict as a JSON line to the session file.

    The file is opened in append mode so existing content is never overwritten.
    Errors are reported but never raised — a persistence failure should not
    crash the agent.
    """
    try:
        path = Path(session_file)
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_msg = _serialisable(message)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(safe_msg) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️  Failed to persist message: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent_turn(
    client: anthropic.Anthropic,
    messages: list,
    system: str | None = None,
) -> str:
    """
    Run one full agent turn (potentially multiple model calls if tool_use is
    involved) and return the final text response.

    Args:
        client:  Anthropic API client.
        messages: Conversation history (mutated in-place on tool loops).
        system:  Optional system prompt string (loaded from SOUL.md).
    """
    while True:
        create_kwargs: dict = dict(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOLS,
            messages=messages,
        )
        if system:
            create_kwargs["system"] = system

        response = client.messages.create(**create_kwargs)

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
    parser = argparse.ArgumentParser(description="Minimal AI agent with persistent conversation memory.")
    parser.add_argument(
        "--session",
        default=os.environ.get("AGENT_SESSION_FILE", DEFAULT_SESSION_FILE),
        metavar="FILE",
        help=f"Path to the JSONL session file (default: {DEFAULT_SESSION_FILE}). "
             "Can also be set via the AGENT_SESSION_FILE environment variable.",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Start a fresh session, ignoring any existing session file.",
    )
    parser.add_argument(
        "--soul",
        metavar="PATH",
        default=None,
        help=(
            "Path to a SOUL.md file whose contents are injected as the system prompt. "
            "Defaults to SOUL.md in the current directory if it exists."
        ),
    )
    args = parser.parse_args()

    # ---- Load soul / system prompt ----
    soul = load_soul(args.soul)

    session_file = args.session

    # ---- Load or reset session ----
    if args.new:
        messages = []
        print(f"🆕 Starting fresh session (ignoring '{session_file}' if it exists).")
    else:
        messages = load_session(session_file)
        if messages:
            print(f"📂 Resumed session from '{session_file}' ({len(messages)} message(s) loaded).")
        else:
            print(f"📂 No previous session found at '{session_file}'. Starting fresh.")

    client = anthropic.Anthropic()

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

        # Snapshot the message list length *before* appending the user message
        # so that if an API error occurs mid-turn we can roll back the entire
        # failed turn (user message + any tool_result entries) in one slice.
        cursor = len(messages)
        user_msg = {"role": "user", "content": user_input}
        messages.append(user_msg)

        try:
            assistant_text = run_agent_turn(client, messages, system=soul)

            # Persist the final assistant text turn in history
            assistant_msg = {"role": "assistant", "content": assistant_text}
            messages.append(assistant_msg)

            # Write both new messages to disk (only after a successful turn)
            append_message(session_file, user_msg)
            append_message(session_file, assistant_msg)

            print(f"\nAgent: {assistant_text}\n")

        except anthropic.APIError as e:
            print(f"\n❌ API error: {e}\n")
            # Roll back every message appended during this turn (user message
            # plus any assistant/tool_result pairs added inside run_agent_turn).
            del messages[cursor:]


if __name__ == "__main__":
    main()
