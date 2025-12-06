import os
import json
import warnings

from loguru import logger

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.slack_integrations_online.application.agents.tools.memory_tools import add_to_memory, search_memory
from src.slack_integrations_online.application.agents.tools.monogdb_retriever_tools import mongodb_retriever_tool, get_complete_docs_with_url

warnings.filterwarnings("ignore", category=DeprecationWarning)


INSTRUCTIONS="""You are a helpful agent that uses tools to answer user queries accurately.

**Step 1: Identify User Intent**
Determine if the user is asking about their previous memories/conversations or asking a new question.

**If user is asking about memories:**
- Use search_memory tool to retrieve relevant memories
- Provide the response based on retrieved memories
- Do NOT use other tools

**If user is asking a new question:**
1. First, use search_memory to check for relevant past context
2. Use mongodb_retriever_tool to search for relevant documents
3. Answer using ONLY information from the retrieved documents
4. If the chunks lack detail, use get_complete_docs_with_url to fetch complete documents
5. Finally, use add_to_memory to store this interaction for future reference

**Guidelines:**
- Be concise and accurate
- Quote relevant parts from documents when appropriate
- If information is not found, say "I don't have enough information to answer this question"
- Always cite document URLs in your final answer at the end, when using information from documents
- Only use get_complete_docs_with_url when chunks are relevant to the query but lack sufficient detail or context
"""

model = ChatOpenAI(model="gpt-4o-mini")
tools = [search_memory, mongodb_retriever_tool, get_complete_docs_with_url, add_to_memory]


def create_agent_graph():
    """Create a LangGraph agent with tools using StateGraph."""
    
    # Bind tools to model
    model_with_tools = model.bind_tools(tools)
    
    # Define the function that calls the model
    def call_model(state: MessagesState):
        messages = state["messages"]
        
        # Add system message if not present or if it's the first message
        if not messages or not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=INSTRUCTIONS)] + messages
        
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Build the graph
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
    )
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


agent_graph = create_agent_graph()

logger.info("Initializing agent with the following tools:")

for tool in tools:
    logger.info(f"Tool name: {tool.name}")
    logger.info(f"Tool description: {tool.description}")
    logger.info(f"Tool parameters: {json.dumps(tool.args, indent=2)}")


class SupportAgentsManager():
    """Manager for running support agents with memory context and trace logging."""
    
    def __init__(self) -> None:
        pass

    async def run(self, query:str, user_id: str = "default_user") -> None:

        try:
                
            logger.info("Starting the agent run")
            inputs = {"messages": [HumanMessage(content=f"User query: {query}")]}
            config = {"configurable": {"user_id": user_id}}

            result = await agent_graph.ainvoke(inputs, config=config)

            final_output = result["messages"][-1].content
            logger.info(f"Agent response: {final_output}")

            return final_output

        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")