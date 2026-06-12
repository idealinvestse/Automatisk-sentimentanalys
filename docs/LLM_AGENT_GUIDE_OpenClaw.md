# LLM Agent Guide for OpenClaw / OpenKlaw

**Note**: This is a tailored guide for your OpenClaw/OpenKlaw agent system (the control hub for azom sites, LLM routing, automation, etc.). It follows the same philosophy as the guide created for Automatisk-sentimentanalys.

> **Primary instruction for all coding agents**: Read this file thoroughly before making changes to the OpenClaw system.

## 1. System Mission & Philosophy

**Mission**: Build a powerful, reliable, self-hosted AI agent system (OpenClaw/OpenKlaw) that can autonomously manage and improve e-commerce sites (WooCommerce), perform analysis, translation, content generation, and complex multi-step tasks — while remaining cost-efficient and controllable.

**Core Principles**:
- **Hybrid LLM routing**: Use RouteLLM + LiteLLM to intelligently route tasks to the best/cheapest/fastest model.
- **Tool-use first**: Strong emphasis on reliable tool calling and structured output.
- **Modular & Extensible**: Easy to add new tools, agents, and capabilities.
- **Observability & Control**: Good logging, state management, and human-in-the-loop options.
- **Cost & Latency awareness**: Prefer cheaper/faster models when quality is sufficient.

## 2. High-Level Architecture (from memory of previous discussions)

```
User / Scheduled Task / API
          ↓
OpenClaw Agent Core (orchestration, memory, planning)
          ↓
LLM Router (RouteLLM + LiteLLM)
          ↓
Tool Registry / Function Calling
          ↓
Specific Tools:
  - WooCommerce API tools (products, orders, content)
  - Translation tools
  - Site analysis / scraping tools
  - Content generation tools
  - Database / state tools
          ↓
Result + Logging + State Update
```

## 3. Key Components to Understand

| Component              | Purpose                                      | Important Files / Concepts                  |
|------------------------|----------------------------------------------|---------------------------------------------|
| LLM Router             | Intelligent model selection & routing        | RouteLLM, LiteLLM config, model fallbacks   |
| Tool System            | Structured tool calling                      | Tool definitions, Pydantic schemas          |
| Agent Core             | Planning, memory, multi-step execution       | Main agent loop, state management           |
| WooCommerce Integration| Site management automation                   | Product sync, content updates, order tools  |
| Evaluation / Logging   | Track performance, cost, success rate        | Logging, cost tracking, success metrics     |

## 4. Development Guidelines for Agents

### Adding a New Tool
1. Define the tool using clear Pydantic models for input/output.
2. Register the tool in the central tool registry.
3. Add good docstrings and examples.
4. Write tests that mock the external service (WooCommerce, etc.).
5. Update the agent prompt/examples if the tool changes behavior significantly.

### Modifying LLM Routing Logic
- Be very careful with cost/latency/quality tradeoffs.
- Prefer structured output (`response_format` / Pydantic) whenever possible.
- Always implement fallback logic if a model fails.
- Log model choice + reason + cost.

### Working with State & Memory
- Prefer explicit state management over relying only on LLM memory.
- Use structured storage (JSON, database) for important state.
- Implement checkpointing for long-running tasks.

## 5. Important Patterns

- **Structured Output First**: Almost all LLM calls should return validated Pydantic models.
- **Defensive Tool Design**: Tools should handle errors gracefully and return clear success/failure information.
- **Cost Awareness**: Log token usage and estimated cost on expensive operations.
- **Idempotency**: Tools that modify external systems (WooCommerce) should be as idempotent as possible.

## 6. Security & Best Practices

- Never hardcode API keys (WooCommerce, OpenAI, etc.).
- Use environment variables or secure secret storage.
- Be careful with scraping / external calls — respect rate limits.
- Log sensitive operations.
- Prefer read-only tools by default when possible.

## 7. Recommended Workflow for LLM Agents

1. Read `LLM_AGENT_GUIDE_OpenClaw.md` + any existing architecture docs.
2. Understand the current tool registry and routing logic.
3. Make the smallest possible change that achieves the goal.
4. Add or update tests.
5. Verify cost/latency impact if relevant.
6. Update documentation if the change affects how the agent should be used.

## 8. Common Pitfalls to Avoid

- Over-engineering tool schemas (keep them focused).
- Forgetting fallback logic in the LLM router.
- Making changes that increase cost significantly without clear benefit.
- Ignoring error handling in tools that call external APIs.
- Changing routing logic without measuring impact on success rate and cost.

## 9. Quick Commands / Testing

(Adapt these to your actual setup)

```bash
# Example commands (adjust to your environment)
python -m openclaw.agent --task "update product descriptions"
# or
python main.py --config configs/production.yaml
```

## 10. Related Documentation

- Main project README
- Architecture docs (if they exist)
- Tool registry documentation
- Prompt templates and system prompts

---

**This guide should be updated whenever the core architecture or important patterns in OpenClaw change.**