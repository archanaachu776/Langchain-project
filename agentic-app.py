import streamlit as st
import os
from typing import TypedDict, Annotated
import operator

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from tavily import TavilyClient

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Research Assistant", page_icon="🔍", layout="wide")

st.title("🔍 Multi-Agent Research Assistant")
st.markdown("This agent uses **LangGraph** to decide whether to answer directly or browse the web via **Tavily**.")

# --- 2. SIDEBAR SETUP (API Keys) ---
with st.sidebar:
    st.header("Configuration")
    groq_key = st.text_input("Groq API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")
    
    st.info("The agent will use Llama-3.3-70b for high-quality reasoning.")
    
    if not groq_key or not tavily_key:
        st.warning("Please enter both API keys to proceed.")

# --- 3. CORE LOGIC (Nodes & Graph) ---
class ResearchState(TypedDict):
    query: str
    needs_search: bool
    search_results: str
    final_answer: str
    steps: Annotated[list[str], operator.add]

def get_graph(groq_api_key, tavily_api_key):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=groq_api_key)
    tavily_client = TavilyClient(api_key=tavily_api_key)

    def analyze_query(state: ResearchState):
        prompt = f"Analyze if this query needs web search or direct answer: {state['query']}\nRespond 'SEARCH' or 'DIRECT'."
        response = llm.invoke(prompt)
        needs_search = "SEARCH" in response.content.upper()
        return {"needs_search": needs_search, "steps": [f"🧠 Decision: {'Web Search' if needs_search else 'Direct Knowledge'}"]}

    def search_web(state: ResearchState):
        search_response = tavily_client.search(query=state["query"], max_results=3)
        results = "\n\n".join([f"Source: {r['url']}\n{r['content']}" for r in search_response['results']])
        return {"search_results": results, "steps": [f"🌐 Found {len(search_response['results'])} sources via Tavily"]}

    def synthesize_answer(state: ResearchState):
        prompt = f"Using these results: {state['search_results']}\n\nAnswer the query: {state['query']}"
        response = llm.invoke(prompt)
        return {"final_answer": response.content, "steps": ["✍️ Synthesized research into final report"]}

    def direct_answer(state: ResearchState):
        response = llm.invoke(f"Answer concisely: {state['query']}")
        return {"final_answer": response.content, "steps": ["💡 Answered from internal knowledge"]}

    def route_query(state: ResearchState):
        return "search" if state["needs_search"] else "direct"

    # Build Graph
    workflow = StateGraph(ResearchState)
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("search", search_web)
    workflow.add_node("synthesize", synthesize_answer)
    workflow.add_node("direct", direct_answer)
    
    workflow.set_entry_point("analyze")
    workflow.add_conditional_edges("analyze", route_query, {"search": "search", "direct": "direct"})
    workflow.add_edge("search", "synthesize")
    workflow.add_edge("synthesize", END)
    workflow.add_edge("direct", END)
    
    return workflow.compile()

# --- 4. USER INTERFACE ---
query = st.text_input("What would you like to research today?", placeholder="e.g., What are the latest updates in AI from Dec 2024?")

if st.button("Run Agent") and groq_key and tavily_key:
    with st.status("Agent is working...", expanded=True) as status:
        # Initialize graph
        app = get_graph(groq_key, tavily_key)
        
        # Initial State
        initial_input = {
            "query": query,
            "needs_search": False,
            "search_results": "",
            "final_answer": "",
            "steps": []
        }
        
        # Execute
        result = app.invoke(initial_input)
        
        # Show steps in the status box
        for step in result["steps"]:
            st.write(step)
        
        status.update(label="Research Complete!", state="complete", expanded=False)

    # Display Results
    st.divider()
    st.subheader("Final Answer")
    st.markdown(result["final_answer"])
    
    # Optional: Show raw sources if search was used
    if result["search_results"]:
        with st.expander("View Raw Search Data"):
            st.text(result["search_results"])

elif not (groq_key and tavily_key) and query:
    st.error("Please provide API keys in the sidebar.")