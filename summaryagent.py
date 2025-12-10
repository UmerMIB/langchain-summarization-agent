from dotenv import load_dotenv
from typing import Callable

from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, SummarizationMiddleware
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI


load_dotenv()

SUMMARY_THRESHOLD = 3  # Number of messages before summarization triggers
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

    msgs = list(request.messages) or []
    if not msgs:
        return handler(request)
    compressed = []

    # If we exceed the threshold, summarize all messages before the threshold
    if len(msgs) > SUMMARY_THRESHOLD:
        # Keep the last SUMMARY_THRESHOLD messages
        messages_to_keep = msgs[-SUMMARY_THRESHOLD:]
        
        # Summarize all messages before those
        messages_to_summarize = msgs[:-SUMMARY_THRESHOLD]
        
        chunk_text = "\n".join(getattr(m, "content", str(m)) for m in messages_to_summarize)
        
        summary_response = summarizer.invoke([
            HumanMessage(content=f"Summarize this conversation briefly in 1-2 sentences:\n{chunk_text}")
        ])
        summary_text = summary_response.content
        
        compressed.append(SystemMessage(content=f"[Summary] {summary_text}"))
        msgs = messages_to_keep

    # Build new message list: summaries + remaining messages
    new_messages = compressed + msgs

    # Call the actual agent model with compressed messages
    try:
        return handler(request.override(messages=new_messages))
    except Exception as e:
        print(f"Error: {e}; retrying with original messages.")
        
agent = create_agent(
    model="gpt-4o-mini",
    middleware=[summary_middleware],
    # middleware=[
    #     SummarizationMiddleware(
    #         model="gpt-4o-mini",
    #         trigger=("tokens", 30),
    #         keep=("messages", 3),
    #     ),
    # ],
)

if __name__ == "__main__":
    messages = []
    turn = 0

    print("=" * 60)
    print("Agent with Summary Middleware")
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

        try:
            # Invoke agent with accumulated messages
            response = agent.invoke({"messages": messages})

            agent_response = response["messages"][-1].content
            
            # Add agent response to history
            messages.append(AIMessage(content=agent_response))
            print(f"\n🤖 Agent: {agent_response}\n")
            print(f"📊 Message count after agent call: {len(messages)}\n")

        except Exception as e:
            print(f"❌ Error: {e}\n")

    print("\nTest completed.")
