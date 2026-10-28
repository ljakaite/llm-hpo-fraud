"""
One-call connection test for the Claude API (c_vs_7b). Costs well under one cent.
Reads the key from ANTHROPIC_API_KEY; never prints it.

    .venv/bin/python -I experiments/hpo_llm/checks/api_hello.py
"""

import anthropic

MODEL_ID, EFFORT = "claude-opus-5-5", "low"   # V4a, protocol section 5

client = anthropic.Anthropic()                 # credentials from the environment
response = client.messages.create(
    model=MODEL_ID,
    max_tokens=1000,
    thinking={"type": "adaptive", "display": "summarized"},
    output_config={"effort": EFFORT},
    messages=[{"role": "user", "content":
               'Reply with one JSON object only: {"learning_rate": <a float between 0.01 and 0.3>}'}],
)
print("SDK", anthropic.__version__, "| model", response.model, "| stop_reason", response.stop_reason)
for block in response.content:
    if block.type == "text":
        print("answer:", block.text.strip())
    elif block.type == "thinking":
        print("thinking summary:", (block.thinking or "")[:200])
u = response.usage
print("tokens: input", u.input_tokens, "| output (incl. thinking)", u.output_tokens)
