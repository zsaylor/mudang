import json
import sys

import requests
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py 'Your initial query here'")
        print("Example: python script.py 'Search for the weather in Seoul'")
        return

    initial_query = sys.argv[1]

    model = init_chat_model("gemini-2.5-flash", model_provider="google_vertexai")

    search = TavilySearch(max_results=2)
    tools = [search, get_todo_item]

    memory = MemorySaver()

    app = create_shakespeare_agent(model, tools, memory)
    config = {"configurable": {"thread_id": "zach123"}}

    print("🎭 Shakespeare Weather Bot - Interactive Mode")
    print("=" * 60)
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("=" * 60)

    current_query = initial_query

    while True:
        print(f"\nUser: {current_query}")
        print("\nShakespeare Weather Bot:")
        print("-" * 50)

        input_message = {"role": "user", "content": current_query}

        try:
            for chunk, metadata in app.stream(
                {"messages": [input_message]}, config, stream_mode="messages"
            ):
                if metadata.get("langgraph_node") == "agent" and isinstance(
                    chunk, AIMessage
                ):
                    if hasattr(chunk, "content") and chunk.content:
                        print(chunk.content, end="", flush=True)
            print("\n" + "-" * 50)

        except Exception as e:
            print(f"An error occurred: {e}")
            print("-" * 50)

        # Get next query from user input
        try:
            current_query = input("\nYour next question (or 'quit' to exit): ").strip()

            # Check for exit commands
            if current_query.lower() in ["quit", "exit", "bye", ""]:
                print("\nFarewell! May the weather be ever in thy favor! 🌤️")
                break

        except KeyboardInterrupt:
            print("\n\nFarewell! May the weather be ever in thy favor! 🌤️")
            break
        except EOFError:
            print("\n\nFarewell! May the weather be ever in thy favor! 🌤️")
            break


def create_shakespeare_agent(model, tools, memory):
    """Create a ReAct agent with Shakespearean personality"""
    system_message = """You are a helpful weather assistant that speaks in Shakespearean English. 
                    Answer all questions to the best of your ability using the tools available to you.
                    Always explain what thou art doing before using tools, and provide clear summaries 
                    after getting results, all in the manner of Shakespeare's tongue."""

    return create_react_agent(model, tools, checkpointer=memory, prompt=system_message)


@tool
def get_todo_item(todo_id: int = 1) -> str:
    """
    Fetch a specific todo item from the JSONPlaceholder API.

    Args:
        todo_id: The ID of the todo item to retrieve (default: 1)

    Returns:
        JSON string with the todo item data
    """

    try:
        response = requests.get(f"https://jsonplaceholder.typicode.com/todos/{todo_id}")
        response.raise_for_status()

        todo_data = response.json()
        return json.dumps(todo_data, indent=2)

    except requests.exceptions.RequestException as e:
        return f"Error fetching todo item: {str(e)}"
    except json.JSONDecodeError as e:
        return f"Error parsing JSON response: {str(e)}"


if __name__ == "__main__":
    main()
