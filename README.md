# LangChain v1 Agent with Message Summarization Middleware

A production-ready example of a LangChain v1 agent that uses a **`wrap_model_call` middleware** to automatically summarize old messages when conversation length exceeds a threshold. This prevents context explosion in long-running conversations while preserving important details.

## Features

- **Scalable conversation memory**: Automatically compresses old messages using an LLM summarizer when message count exceeds a configurable threshold.
- **Smart summarization**: Uses a separate `gpt-4o-mini` model to create contextual summaries, avoiding information loss from simple truncation.
- **Interactive CLI**: Type messages, see the agent respond, and track message compression in real-time.
- **Production-ready**: Includes error handling, fallbacks, and comprehensive logging.

## Quick Start

### 1. Clone and set up virtual environment

```bash
cd /home/umer-mib/Documents/langchain
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

Or export directly in your shell:

```bash
export OPENAI_API_KEY="sk-..."
```

### 4. Run the agent

```bash
python summaryagent.py
```

Type messages at the prompt. The agent will respond and track message count. When messages exceed the threshold (default: 6), the middleware automatically summarizes old turns using an LLM.

Example interaction:

```
[Turn 1] You: What is machine learning?
📊 Message count before agent call: 1
🤖 Agent: Machine learning is a subset of artificial intelligence...

[Turn 2] You: Tell me about supervised learning
📊 Message count before agent call: 3
🤖 Agent: Supervised learning is a type of machine learning...

[Turn 6] You: More examples?
📊 Message count before agent call: 11
⚠️  Middleware will summarize using LLM (threshold: 6 messages)
🤖 Agent: [response with compressed history]
```

## How the Middleware Works

### The `@wrap_model_call` Decorator

The `summary_middleware` function uses the `@wrap_model_call` hook, which wraps **every model call** in the agent. This allows us to:

1. **Intercept messages** before they reach the main agent model (gpt-4o).
2. **Compress old messages** when count exceeds `SUMMARY_THRESHOLD` using a separate LLM (gpt-4o-mini).
3. **Pass compressed messages** to the agent via `handler(request.override(messages=new_messages))`.

### Why `wrap_model_call` instead of other hooks?

| Hook              | When it runs               | Use case                                                          |
| ----------------- | -------------------------- | ----------------------------------------------------------------- |
| `before_model`    | Before each model call     | Lightweight mutations, read-only                                  |
| `wrap_model_call` | **Around** each model call | Full control; can replace request, retry, or call external models |
| `after_model`     | After model returns        | Post-processing responses                                         |
| `before_agent`    | Once, at start             | Setup (runs once per agent invocation)                            |
| `after_agent`     | Once, at end               | Cleanup (runs once per agent invocation)                          |
| `wrap_tool_call`  | Around tool execution      | Intercept tool calls (not model calls)                            |

We chose `wrap_model_call` because:

- We need to **modify the request** (replace messages) before the model sees them.
- We call an external LLM (summarizer) inside the middleware.
- We get full control over success/failure/retry behavior.

### Compression Algorithm

```
While message_count > SUMMARY_THRESHOLD:
  1. Extract oldest CHUNK_SIZE messages
  2. Call summarizer LLM to create a 1-2 sentence summary
  3. Replace chunk with a SystemMessage containing the summary
  4. Repeat until message_count ≤ SUMMARY_THRESHOLD
```

This keeps the conversation bounded and prevents token limit issues.

## Configuration

Edit the top of `summaryagent.py` to tune behavior:

```python
SUMMARY_THRESHOLD = 6      # Start compressing when message count exceeds this
CHUNK_SIZE = 4             # Number of oldest messages to compress in each pass
```

Or set via environment variables:

```bash
export SUMMARY_THRESHOLD=20
export SUMMARY_CHUNK_SIZE=10
python summaryagent.py
```

## Project Files

- **`summaryagent.py`** — Main agent script with `summary_middleware` and interactive CLI loop
- **`requirements.txt`** — Dependencies (langchain, langchain-openai, python-dotenv)
- **`.env.example`** — Example environment configuration
- **`.gitignore`** — Git ignore rules for venv, logs, .env, etc.
- **`README.md`** — This file

## Architecture

```
summaryagent.py
├── summary_middleware (@wrap_model_call)
│   ├── Extracts messages from ModelRequest
│   ├── If count > SUMMARY_THRESHOLD:
│   │   ├── Calls summarizer LLM (gpt-4o-mini) on old chunks
│   │   └── Builds compressed message list
│   └── Calls handler(request.override(messages=...))
│
└── agent (via create_agent)
    ├── Model: gpt-4o
    ├── Middleware: [summary_middleware]
    └── Messages: [compressed or original]
```

## Extending the Middleware

### Add retry logic

```python
@wrap_model_call
def retry_middleware(request: ModelRequest, handler):
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")
```

### Stack multiple middlewares

```python
agent = create_agent(
    model="gpt-4o",
    middleware=[log_middleware, summary_middleware, retry_middleware],
)
```

### Use different summarizer models

Change the summarizer model in `summaryagent.py`:

```python
summarizer = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)  # faster, cheaper
# or
summarizer = ChatOpenAI(model="claude-3-haiku-20240307")  # different provider
```

## Troubleshooting

### ImportError: cannot import ChatOpenAI

```bash
pip install langchain-openai
```

### "OPENAI_API_KEY environment variable is not set"

```bash
export OPENAI_API_KEY="sk-..."
python summaryagent.py
```

Or set it in `.env`:

```
OPENAI_API_KEY=sk-...
```

### Summarizer is too slow

Use a faster model:

```python
summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # already default
# or
summarizer = ChatOpenAI(model="gpt-3.5-turbo")
```

## References

- [LangChain Agents Docs](https://python.langchain.com/docs/modules/agents/)
- [Middleware in LangChain](https://python.langchain.com/docs/concepts/agents)
- [OpenAI API Docs](https://platform.openai.com/docs/api-reference)

This repository contains a minimal LangChain v1-style agent with a simple
middleware that summarizes user input before the agent acts.

Quick start

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI key (required):

```bash
export OPENAI_API_KEY="sk-..."
# or copy to .env and let python-dotenv load it
```

Run the REPL

```bash
python main.py
```

Type messages at the prompt. Type `quit` or `exit` to stop.

Files

- `main.py`: CLI REPL
- `agent/middleware.py`: `SummarizeMiddleware` using langchain/OpenAI
- `agent/agent_builder.py`: `AgentWrapper` that builds a minimal langchain agent
- `requirements.txt`: dependencies

Notes

- This simplified example assumes you will run it with a real OpenAI key
  and the `langchain` package installed.
- If you want a dummy/offline mode preserved, I can add it back as an
  optional flag; for now the code is intentionally short and focused.
