#!/usr/bin/env python3
"""A minimal AI agent: REPL that talks to Claude with conversation memory."""

import sys
import anthropic

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096


def main():
    client = anthropic.Anthropic()
    messages = []

    print("🤖 Agent ready. Type 'quit' to exit.\n")

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
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=messages,
            )

            assistant_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    assistant_text += block.text

            messages.append({"role": "assistant", "content": assistant_text})
            print(f"\nAgent: {assistant_text}\n")

        except anthropic.APIError as e:
            print(f"\n❌ API error: {e}\n")
            # Remove the failed user message so conversation stays valid
            messages.pop()


if __name__ == "__main__":
    main()
