from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from typing import Annotated, List
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
import json
import os


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def load_finance_data():
    # Get the absolute path to the finance_data.json file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    finance_data_path = os.path.join(current_dir, "finance_data.json")

    # Load the finance data from the JSON file
    with open(finance_data_path, "r") as file:
        finance_data = json.load(file)

    return finance_data


@tool
def get_finance_data():
    """
    This tool returns a dataset of company contracts containing information about:
    - Company names
    - Contract amounts
    - Contract lengths (in months)
    - Renewal dates
    """
    return load_finance_data()


@tool
def multiply_by_pi(number: int):
    """
    This tool multiplies a number by pi.
    """
    return 3.14159 * number


llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    betas=["interleaved-thinking-2025-05-14"],
)
finance_tools = [get_finance_data, multiply_by_pi]
model_with_tools = llm.bind_tools(finance_tools)
finance_tool_node = ToolNode(finance_tools)


def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]

    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


def finance_assistant(state: State):
    # Extract the user's question from the last human message
    messages = state["messages"]
    # Find the last human message
    user_question = ""
    for message in reversed(messages):
        if message.type == "human":
            user_question = message.content
            break

    # Create a finance expert prompt
    finance_expert_prompt = """You are an expert financial analyst with deep knowledge of business contracts and financial planning.
    You have access to a dataset of company contracts containing information. If you need to access the data, use the get_finance_data tool, if you need to multiply a number by pi, use the multiply_by_pi tool. If not, answer the user directly.
    It will provide the following information:

    - Company names
    - Contract amounts
    - Contract lengths (in months)
    - Renewal dates
    
    Please analyze this data to provide helpful, accurate, and insightful answers to the user's questions.
    
    The user's question is: {user_question}
    """

    # Call the model with the finance expert prompt and context
    response = model_with_tools.invoke(
        [SystemMessage(finance_expert_prompt)] + state["messages"]
    )

    # Return the answer
    return {"messages": [response]}


# Build the graph with explicit schemas
builder = StateGraph(State)
builder.add_node("finance_assistant", finance_assistant)
builder.add_node("finance_tool_node", finance_tool_node)
builder.add_edge(START, "finance_assistant")
builder.add_conditional_edges(
    "finance_assistant",
    should_continue,
    {
        "continue": "finance_tool_node",
        "end": END,
    },
)
builder.add_edge("finance_tool_node", "finance_assistant")
builder.add_edge("finance_tool_node", END)
graph = builder.compile()

if __name__ == "__main__":
    graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Who are the top 3 contracts by size? Multiply the top by pi"
                )
            ]
        }
    )
