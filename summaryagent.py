from dotenv import load_dotenv
from typing import Callable

from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, SummarizationMiddleware
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

# Load environment variables (like API keys) from .env file
load_dotenv()

# How many recent messages to keep before we start summarizing old ones
# If we have more than 3 messages, we'll summarize the old ones
SUMMARY_THRESHOLD = 3

# Create a separate AI model just for summarizing conversations
# We use a cheaper model (gpt-4o-mini) since summarization doesn't need the best model
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
    
    # Get all messages from the request
    msgs = list(request.messages) or []
    
    # If there are no messages, just pass the request through without changes
    if not msgs:
        return handler(request)
    
    # This list will hold any summary messages we create
    compressed = []

    # Check if we have too many messages (more than our threshold)
    if len(msgs) > SUMMARY_THRESHOLD:
        # Keep only the most recent messages (the last 3 messages)
        # These are the most important for the current conversation
        messages_to_keep = msgs[-SUMMARY_THRESHOLD:]

        # Get all the old messages that came before the recent ones
        # These are the ones we want to summarize
        messages_to_summarize = msgs[:-SUMMARY_THRESHOLD]
        
        # Combine all the old messages into one text string
        # This makes it easier to send to the summarizer
        chunk_text = "\n".join(getattr(m, "content", str(m)) for m in messages_to_summarize)
        
        # Ask the summarizer AI to create a short summary of all the old messages
        summary_response = summarizer.invoke([
            HumanMessage(content=f"Summarize this conversation briefly in 1-2 sentences:\n{chunk_text}")
        ])
        summary_text = summary_response.content
        
        # Create a system message with the summary and add it to our compressed list
        compressed.append(SystemMessage(content=f"[Summary] {summary_text}"))
        
        # Replace the old messages with just the recent ones we're keeping
        msgs = messages_to_keep

    # Build the final message list: summary (if any) + recent messages
    # This way the agent gets context without too many old messages
    new_messages = compressed + msgs

    # Send the compressed messages to the actual agent
    try:
        return handler(request.override(messages=new_messages))
    except Exception as e:
        # If something goes wrong, print the error
        # Note: The code should probably return handler(request) here to retry with original messages
        print(f"Error: {e}; retrying with original messages.")
        
# Create an AI agent that uses our summary middleware
# The middleware will automatically compress old messages before the agent sees them
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
    # This list stores the entire conversation history
    messages = []
    turn = 0

    print("=" * 60)
    print("Agent with Summary Middleware")
    print("=" * 60)
    print("Type 'quit' to exit.\n")

    # Main conversation loop - keeps running until user types 'quit'
    while True:
        turn += 1
        # Ask the user for input
        user_input = input(f"[Turn {turn}] You: ").strip()

        # Check if user wants to exit
        if user_input.lower() == "quit":
            print("Exiting...")
            break

        # Skip empty inputs
        if not user_input:
            continue

        # Add the user's message to our conversation history
        messages.append(HumanMessage(content=user_input))
        
        # Show how many messages we have before calling the agent
        print(f"\n📊 Message count before agent call: {len(messages)}")

        try:
            # Send all messages to the agent
            # The middleware will automatically compress old messages if needed
            response = agent.invoke({"messages": messages})

            # Get the agent's response (the last message in the response)
            agent_response = response["messages"][-1].content
            
            # Add the agent's response to our conversation history
            messages.append(AIMessage(content=agent_response))
            
            # Show the agent's response and current message count
            print(f"\n🤖 Agent: {agent_response}\n")
            print(f"📊 Message count after agent call: {len(messages)}\n")

        except Exception as e:
            # If something goes wrong, show the error
            print(f"❌ Error: {e}\n")

    print("\nTest completed.")


# Summary Middleware Agent
# - My name is Ali, I play cricket everyday in morning 
# - I am kinda jolly boy and loves cracking jokes 
# - I do software development to develop GenAI apps
# - My headphones are missing since joining 
# - Sitting under AC during winter can cause fever