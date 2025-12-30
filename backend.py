import streamlit as st
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from tavily import TavilyClient
import operator

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

# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 AI Research Assistant")
    st.markdown("Ask any question and let the AI determine whether to search the web or answer directly!")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This research assistant uses:
        - **LangGraph** for workflow orchestration
        - **Groq LLM** for intelligent reasoning
        - **Tavily API** for web search
        
        The system automatically decides whether your query needs:
        - 🌐 **Web Search** (for current events/data)
        - 📚 **Direct Answer** (for general knowledge)
        """)
        
        st.divider()
        
        st.header("Example Queries")
        st.markdown("""
        **General Knowledge:**
        - What is the capital of France?
        - Explain quantum computing
        
        **Current Events:**
        - Latest AI developments
        - Recent tech news
        """)
    
    # Main content
    graph = create_graph()
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the latest AI developments?",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🚀 Search", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()
    
    if search_button and query:
        with st.spinner("Processing your query..."):
            try:
                # Execute the graph
                result = graph.invoke({
                    "query": query,
                    "needs_search": False,
                    "search_results": "",
                    "final_answer": "",
                    "steps": []
                })
                
                # Display results
                st.success("✅ Research Complete!")
                
                # Process steps
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔄 Process Steps")
                    for i, step in enumerate(result["steps"], 1):
                        st.markdown(f"**{i}.** {step}")
                    
                    # Show search indicator
                    if result["needs_search"]:
                        st.info("🌐 Used web search for current information")
                    else:
                        st.info("📚 Answered from knowledge base")
                
                with col2:
                    st.subheader("📊 Query Analysis")
                    st.metric("Search Required", "Yes" if result["needs_search"] else "No")
                    st.metric("Steps Executed", len(result["steps"]))
                
                # Final answer
                st.divider()
                st.subheader("💡 Answer")
                st.markdown(result["final_answer"])
                
                # Show search results if available
                if result.get("search_results"):
                    with st.expander("🔍 View Search Results"):
                        st.text(result["search_results"])
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Please check your API keys in the .env file")
    
    elif search_button and not query:
        st.warning("⚠️ Please enter a question first!")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        Built with LangGraph, Groq, and Tavily | Powered by Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()