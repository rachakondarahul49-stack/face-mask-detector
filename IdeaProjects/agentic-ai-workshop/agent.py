from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, MessagesState


# ============================================================
# 1. Tools
# ============================================================

search_tool = TavilySearchResults(max_results=3)

python_tool = PythonREPLTool()

tools = [search_tool, python_tool]


# ============================================================
# 2. Gemini Model
# ============================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
).bind_tools(tools)


# ============================================================
# 3. Agent Node
# ============================================================

def agent_node(state: MessagesState) -> Dict[str, Any]:
    messages = state["messages"]

    response = model.invoke(messages)

    return {"messages": [response]}


# ============================================================
# 4. LangGraph Workflow
# ============================================================

workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent_node)

workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition
)

workflow.add_edge("tools", "agent")

app = workflow.compile()


# ============================================================
# 5. Interactive CLI
# ============================================================

if __name__ == "__main__":

    print("====================================================")
    print("Welcome to your local Agentic AI CLI!")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("====================================================")

    while True:

        try:

            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            events = app.stream(
                {"messages": [("user", user_input)]},
                stream_mode="values"
            )

            last_message = None

            for event in events:

                if "messages" in event:
                    last_message = event["messages"][-1]

            if last_message:
                print(f"\nAgent: {last_message.content}")

        except KeyboardInterrupt:

            print("\nGoodbye!")
            break

        except Exception as e:

            print(f"\nAn error occurred: {e}")