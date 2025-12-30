import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, MessageState, START, END
from langchain_groq import ChatGroq
from tavily import TavilyClient
import operator 



load_dotenv() #load environment variables
llm = llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature = 0,
)

travily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

#define state

class ResearchState(TypedDict):
    query: str
    needs_search: bool
    search_results: str
    final_answer: str
    steps: Annotated[list[str], operator.add]

# Node 1: Analyze Query
def analyze_query(state: ResearchState) -> ResearchState:
    """Determine if the query needs web search"""
    query = state["query"]
    
    prompt = f"""Analyze this query and determine if it needs current web search or can be answered from general knowledge.

Query: {query}

Respond with only 'SEARCH' or 'DIRECT'.
- SEARCH: If it requires recent information, current events, or specific data
- DIRECT: If it's a general knowledge question that doesn't need real-time data

Response:"""
    
    response = llm.invoke(prompt)
    needs_search = "SEARCH" in response.content.upper()
    
    return {
        **state,
        "needs_search": needs_search,
        "steps": [f"Analyzed query - Needs search: {needs_search}"]
    }

# Node 2: Search Web
def search_web(state: ResearchState) -> ResearchState:
    """Search the web using Tavily"""
    query = state["query"]
    
    search_response = tavily_client.search(
        query=query,
        max_results=3
    )
    
    # Format results
    results = "\n\n".join([
        f"Source: {result['url']}\n{result['content']}"
        for result in search_response['results']
    ])
    
    return {
        **state,
        "search_results": results,
        "steps": [f"Searched web - Found {len(search_response['results'])} sources"]
    }

# Node 3: Synthesize with Search
def synthesize_answer(state: ResearchState) -> ResearchState:
    """Create answer using search results"""
    query = state["query"]
    search_results = state["search_results"]
    
    prompt = f"""Based on the following search results, provide a comprehensive answer to the query.

Query: {query}

Search Results:
{search_results}

Provide a well-structured answer with citations where appropriate."""
    
    response = llm.invoke(prompt)
    
    return {
        **state,
        "final_answer": response.content,
        "steps": ["Synthesized answer from search results"]
    }

# Node 4: Direct Answer
def direct_answer(state: ResearchState) -> ResearchState:
    """Answer directly without search"""
    query = state["query"]
    
    prompt = f"""Provide a clear and concise answer to this query based on your knowledge:

Query: {query}


Answer:"""
    
    response = llm.invoke(prompt)
    
    return {
        **state,
        "final_answer": response.content,
        "steps": ["Provided direct answer from knowledge base"]
    }

# Router Function
def route_query(state: ResearchState) -> str:
    """Route to search or direct answer"""
    if state["needs_search"]:
        return "search"
    else:
        return "direct"

# Build the Graph
def create_graph():
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("search", search_web)
    workflow.add_node("synthesize", synthesize_answer)
    workflow.add_node("direct", direct_answer)
    
    # Add edges
    workflow.set_entry_point("analyze")
    
    # Conditional routing after analysis
    workflow.add_conditional_edges(
        "analyze",
        route_query,
        {
            "search": "search",
            "direct": "direct"
        }
    )
    
    # Search path
    workflow.add_edge("search", "synthesize")
    workflow.add_edge("synthesize", END)
    
    # Direct path
    workflow.add_edge("direct", END)
    
    return workflow.compile()

# Main execution
def main():
    graph = create_graph()
    
    # Test queries
    queries = [
        "What is the capital of France?",
        "What are the latest developments in AI in December 2024?",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = graph.invoke({
            "query": query,
            "needs_search": False,
            "search_results": "",
            "final_answer": "",
            "steps": []
        })
        
        print(f"\nSteps taken:")
        for step in result["steps"]:
            print(f"  - {step}")
        
        print(f"\nFinal Answer:\n{result['final_answer']}")

if __name__ == "__main__":
    main()
