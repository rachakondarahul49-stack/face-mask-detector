import os
from typing import Dict, Any

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_experimental.tools import PythonREPLTool

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, MessagesState


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing from your .env file"
    )

if not TAVILY_API_KEY:
    raise ValueError(
        "TAVILY_API_KEY is missing from your .env file"
    )


# ============================================================
# 2. CREATE TOOLS
# ============================================================

# Tavily web search
search_tool = TavilySearch(
    max_results=3,
    tavily_api_key=TAVILY_API_KEY
)


# Python calculator / execution
python_tool = PythonREPLTool()


# Put all tools into one list
tools = [
    search_tool,
    python_tool
]


# ============================================================
# 3. GEMINI MODEL
# ============================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
).bind_tools(tools)


# ============================================================
# 4. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an intelligent Agentic AI Assistant.

You have access to two tools.

============================================================
TOOL 1: TAVILY WEB SEARCH
============================================================

Use Tavily Web Search whenever the user asks for information
that may require current or real-time information.

Examples:

- latest information
- current information
- today's information
- recent events
- news
- sports results
- Formula 1 results
- latest AI developments
- current technology news
- current prices
- live internet information
- recent company information
- anything that requires browsing the internet

IMPORTANT:

Never answer a "latest", "current", "today", "recent", or
"who won" question using your own knowledge.

Use Tavily first.

============================================================
TOOL 2: PYTHON
============================================================

Use Python for:

- calculations
- mathematical operations
- numerical problems
- statistics
- percentages
- averages
- large calculations
- data manipulation

For simple calculations, Python is preferred.

============================================================
FINAL RESPONSE
============================================================

After using a tool:

1. Understand the tool result.
2. Give the user a clear answer.
3. Do not expose internal tool calls.
4. Do not say that you cannot browse if Tavily is available.
5. If the search results are uncertain, clearly mention that.
"""


# ============================================================
# 5. AGENT NODE
# ============================================================

def agent_node(state: MessagesState) -> Dict[str, Any]:

    system_message = SystemMessage(
        content=SYSTEM_PROMPT
    )

    response = model.invoke(
        [system_message] + state["messages"]
    )

    return {
        "messages": [response]
    }


# ============================================================
# 6. CREATE LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(MessagesState)


# Add agent node
workflow.add_node(
    "agent",
    agent_node
)


# Add tools node
workflow.add_node(
    "tools",
    ToolNode(tools)
)


# Start -> Agent
workflow.add_edge(
    START,
    "agent"
)


# Agent -> Tools OR End
workflow.add_conditional_edges(
    "agent",
    tools_condition
)


# Tools -> Agent
workflow.add_edge(
    "tools",
    "agent"
)


# Compile graph
app = workflow.compile()


# ============================================================
# 7. PRINT RESPONSE CLEANLY
# ============================================================

def print_clean_response(content):

    if isinstance(content, str):

        print(content)

    elif isinstance(content, list):

        for item in content:

            if isinstance(item, dict):

                text = item.get("text")

                if text:
                    print(text)

            else:
                print(item)

    else:

        print(content)


# ============================================================
# 8. CHAT APPLICATION
# ============================================================

def main():

    print("=" * 65)
    print("                 AGENTIC AI ASSISTANT")
    print("=" * 65)

    print()
    print("Available capabilities:")
    print("  • Tavily Web Search")
    print("  • Python Calculator")
    print("  • Gemini AI")
    print("  • LangGraph Agent")
    print()

    print("Examples:")
    print("  Calculate 25 * 40")
    print("  Who won the latest Formula 1 Grand Prix?")
    print("  What are the latest AI developments?")
    print("  Explain machine learning")
    print()

    print("Type 'exit' to stop.")
    print("=" * 65)


    while True:

        try:

            user_input = input("\nYou: ").strip()


            # Ignore empty input
            if not user_input:
                continue


            # Exit commands
            if user_input.lower() in [
                "exit",
                "quit",
                "q"
            ]:

                print("\nGoodbye!")
                break


            # ==================================================
            # RUN LANGGRAPH
            # ==================================================

            result = app.invoke(
                {
                    "messages": [
                        ("user", user_input)
                    ]
                }
            )


            # ==================================================
            # GET FINAL MESSAGE
            # ==================================================

            final_message = result["messages"][-1]


            print("\nAgent:")

            print_clean_response(
                final_message.content
            )


        except KeyboardInterrupt:

            print("\n\nGoodbye!")
            break


        except Exception as e:

            print("\nError:")
            print(e)


# ============================================================
# 9. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()