import os
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from tavily import TavilyClient
import operator

# Load environment variables
load_dotenv()

# Initialize LLM and Tavily client
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )

@st.cache_resource
def get_tavily_client():
    return TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

llm = get_llm()
tavily_client = get_tavily_client()

# Define state
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
@st.cache_resource
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

# Streamlit App
def main():
    st.set_page_config(
        page_title="AI Research Agent",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 AI Research Agent")
    st.markdown("Ask any question and let the agent decide whether to search the web or answer from knowledge.")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This agent uses LangGraph to:
        1. **Analyze** your query
        2. **Route** to web search or direct answer
        3. **Synthesize** results into a comprehensive response
        
        **Powered by:**
        - Groq (Llama 3.1)
        - Tavily Search API
        - LangGraph
        """)
        
        st.divider()
        
        st.header("Example Queries")
        example_queries = [
            "What is the capital of France?",
            "What are the latest developments in AI?",
            "Explain quantum computing",
            "What happened in the news today?"
        ]
        
        for example in example_queries:
            if st.button(example, key=example):
                st.session_state.query_input = example
    
    # Initialize session state
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""
    
    # Main input
    query = st.text_input(
        "Enter your question:",
        value=st.session_state.query_input,
        placeholder="e.g., What are the latest AI breakthroughs?",
        key="main_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("🚀 Ask", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.query_input = ""
        st.rerun()
    
    if submit_button and query:
        with st.spinner("🤔 Thinking..."):
            try:
                # Create graph
                graph = create_graph()
                
                # Execute
                result = graph.invoke({
                    "query": query,
                    "needs_search": False,
                    "search_results": "",
                    "final_answer": "",
                    "steps": []
                })
                
                # Display results
                st.success("✅ Complete!")
                
                # Show reasoning steps
                with st.expander("🔄 Reasoning Steps", expanded=True):
                    for i, step in enumerate(result["steps"], 1):
                        if "Needs search: True" in step:
                            st.info(f"**Step {i}:** {step}")
                        elif "Needs search: False" in step:
                            st.warning(f"**Step {i}:** {step}")
                        else:
                            st.success(f"**Step {i}:** {step}")
                
                # Show search results if available
                if result.get("search_results"):
                    with st.expander("📚 Search Results"):
                        st.text(result["search_results"])
                
                # Show final answer
                st.markdown("### 💡 Answer")
                st.markdown(result["final_answer"])
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Make sure your API keys are set in the .env file")

if __name__ == "__main__":
    main()