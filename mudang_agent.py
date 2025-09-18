# https://python.langchain.com/docs/tutorials/agents/

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

search = TavilySearch(max_results=2)
tools = [search]

# SIMPLE DIRECT INVOKE OF TOOL
# search_results = search.invoke("What is the weather in Seoul?")
# print("search_results:")
# print(search_results)

model = init_chat_model("gemini-2.5-flash", model_provider="google_vertexai")
# Note: Gemini seems to be inable to send a message along with tool call w/out explicit prompting
query = "Search for the weather in Seoul."
config = {"configurable": {"thread_id": "abc123"}}
memory = MemorySaver()

# SIMPLE MODEL RESPONSE
# response = model.invoke([{"role": "user", "content": query}])
# print("response.text():")
# response.text()

# SIMPLE MODEL WITH TOOLS
# model_with_tools = model.bind_tools(tools)
# response = model_with_tools.invoke([{"role": "user", "content": query}])
# print(f"Message content: {response.text()}\n")
# print(f"Tool calls: {response.tool_calls}")

agent_executor = create_react_agent(model, tools, checkpointer=memory)
input_message = {"role": "user", "content": query}

# NON STREAMED OUTPUT
# response = agent_executor.invoke(
#     {
#         "messages": [input_message],
#         "prompt": "You are a helpful weather assistant. Always explain what you're doing before using tools, and provide clear summaries after getting results.",
#     }
# )
# for message in response["messages"]:
#     message.pretty_print()

# STREAMING MULTIPLE AGENT STEPS
# for step in agent_executor.stream({"messages": [input_message]}, stream_mode="values"):
#     step["messages"][-1].pretty_print()

# STREAMING TOKENS
for step, metadata in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="messages"
):
    if metadata["langgraph_node"] == "agent" and (text := step.text()):
        print(text, end="|")

# MEMORY
# for step in agent_executor.stream(
#     {"messages": [("user", "Hi, I'm Bob!")]}, config, stream_mode="values"
# ):
#     step["messages"][-1].pretty_print()
#
# response = agent_executor.invoke({"messages": [("user", "What's my name?")]}, config)
# for message in response["messages"]:
#     message.pretty_print()
