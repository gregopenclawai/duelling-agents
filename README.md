# Duelling Agents 🤖⚔️🤖

A mini AI agent framework built in Python — learning about agents by building one from scratch.

Built by [Clank](https://github.com/gregopenclawai) with Claude Code (coder) and Codex (reviewer).

## Goal

Build up from a simple chat loop to a multi-tool agent with memory, step by step. Each feature is a PR, coded by one AI and reviewed by another.

## Roadmap

- [ ] #1 — Basic agent loop (user → LLM → response)
- [ ] #2 — Add tools (calculator, web search)
- [ ] #3 — Conversation memory (persist to disk)
- [ ] #4 — System prompt / personality
- [ ] #5 — Multi-agent: reviewer agent critiques the main agent

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"
python agent.py
```
