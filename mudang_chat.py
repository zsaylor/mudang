# https://python.langchain.com/docs/tutorials/chatbot/

from typing import Sequence

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

model = init_chat_model("gemini-2.5-flash", model_provider="google_vertexai")

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk in Shakespearean English. Answer all questions to the best of your ability. Speak like a: {station}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    station: str


# Define a new graph
workflow = StateGraph(state_schema=State)

# TODO: find best max tokens value
trimmer = trim_messages(
    max_tokens=650,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)


# Define the function that calls the model
def call_model(state: State):
    # print(f"Messages before trimming: {len(state['messages'])}")
    trimmed_messages = trimmer.invoke(state["messages"])
    # print(f"Messages after trimming: {len(trimmed_messages)}")
    # print("Remaining messages:")
    # for msg in trimmed_messages:
    #     print(f"  {type(msg).__name__}: {msg.content}")
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "station": state["station"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "zach123"}}

query = "Hi! I'm Zach."
station = "jilted lover"
input_messages = [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"messages": input_messages, "station": station},
    config,
    stream_mode="messages",
):
    if isinstance(chunk, AIMessage):  # Filter to just model responses
        print(chunk.content, end="|")
# output = app.invoke(
#     {"messages": input_messages, "station": station},
#     config,
# )
# output["messages"][-1].pretty_print()

# Note: no need to put station again because entire state is persisted.
# query = "Who are you?"
# input_messages = [HumanMessage(query)]
# output = app.invoke(
#     {"messages": input_messages},
#     config,
# )
# output["messages"][-1].pretty_print()
