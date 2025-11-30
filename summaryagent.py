import os
import sys
from dotenv import load_dotenv

from langchain.agents.middleware import wrap_model_call
from langchain.agents.middleware import  ModelRequest, ModelResponse
from langchain.agents import create_agent
from typing import Callable
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

load_dotenv()

SUMMARY_THRESHOLD = 6  # Number of messages before summarization triggers
CHUNK_SIZE = 4        # Number of messages to compress at once

# Direct summarizer LLM (separate from agent, so no recursion)
summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)


@wrap_model_call
def summary_middleware(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """
    Middleware that compresses old messages using an LLM summarizer.
    Calls a separate summarizer model to preserve context, then passes compressed messages to agent.
    """

    # Extract messages from request
    messages = None
    if hasattr(request, "messages"):
        messages = list(request.messages) or []
    elif isinstance(getattr(request, "kwargs", None), dict):
        messages = list(request.kwargs.get("messages") or [])
    else:
        messages = []

    # If under threshold, pass through unchanged
    if len(messages) <= SUMMARY_THRESHOLD:
        return handler(request)

    # Compress old messages in chunks
    msgs = messages[:]
    compressed = []

    while len(msgs) > SUMMARY_THRESHOLD:
        chunk = msgs[:CHUNK_SIZE]
        chunk_text = "\n".join(getattr(m, "content", str(m)) for m in chunk)
        
        # Call summarizer LLM to produce a concise summary
        try:
            summary_response = summarizer.invoke([
                HumanMessage(content=f"Summarize this conversation briefly in 1-2 sentences:\n{chunk_text}")
            ])
            summary_text = summary_response.content
        except Exception as e:
            print(f"Summarizer error: {e}; falling back to truncation.")
            summary_text = chunk_text[:300] + "…"
        
        compressed.append(SystemMessage(content=f"[Summary] {summary_text}"))
        msgs = msgs[CHUNK_SIZE:]

    # Build new message list: summaries + remaining messages
    new_messages = compressed + msgs

    print(f"🗜️ Compressed {len(messages)} → {len(new_messages)} messages via summarization.")
    print(f"🗜️ Compressed Messages: {new_messages}")

    # Call the actual agent model with compressed messages
    try:
        return handler(request.override(messages=new_messages))
    except Exception as e:
        print(f"Error: {e}; retrying with original messages.")
        for attempt in range(2):
            try:
                return handler(request)
            except Exception as e2:
                if attempt == 1:
                    raise
                print(f"Retry {attempt + 1}/2")

agent = create_agent(
    model="gpt-4o",
    middleware=[summary_middleware],
)

if __name__ == "__main__":
    messages = []
    turn = 0

    print("=" * 60)
    print("Agent with Summary Middleware - Interactive Test")
    print("=" * 60)
    print("Type 'quit' to exit.\n")

    while True:
        turn += 1
        user_input = input(f"[Turn {turn}] You: ").strip()

        if user_input.lower() == "quit":
            print("Exiting...")
            break

        if not user_input:
            continue

        # Add user message to history
        messages.append(HumanMessage(content=user_input))
        
        print(f"\n📊 Message count before agent call: {len(messages)}")
        if len(messages) > 5:
            print(f"⚠️  Middleware will summarize (threshold: 5 messages)\n")

        try:
            # Invoke agent with accumulated messages
            response = agent.invoke({"messages": messages})
            
            # Extract response text
            if isinstance(response, dict) and "output" in response:
                agent_response = response["output"]
            else:
                agent_response = str(response)
            
            # print(f"\n🤖 Agent: {agent_response}\n")
            
            # Add agent response to history
            messages.append(AIMessage(content=agent_response))
            print(f"📊 Message count after agent call: {len(messages)}\n")

        except Exception as e:
            print(f"❌ Error: {e}\n")

    print("\nTest completed.")