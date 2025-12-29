"""
Streamlit UI for Multi-Agent AI System.

Provides an interactive web interface for interacting with
the multi-agent system.
"""

import streamlit as st
import requests
import json
import time
from typing import Optional, Dict, Any


# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "selected_agents" not in st.session_state:
        st.session_state.selected_agents = []


def get_agents() -> list:
    """Fetch available agents from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/agents")
        if response.status_code == 200:
            return response.json().get("agents", [])
    except Exception as e:
        st.error(f"Failed to fetch agents: {e}")
    return []


def execute_query(query: str, agents: list = None) -> Dict[str, Any]:
    """Execute a query through the API."""
    try:
        payload = {
            "query": query,
            "agents": agents,
            "output_format": "detailed"
        }
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Agent selection
        st.subheader("Select Agents")
        agents = get_agents()
        
        if agents:
            for agent in agents:
                if st.checkbox(
                    f"{agent['name'].title()}",
                    value=agent['name'] in st.session_state.selected_agents,
                    key=f"agent_{agent['name']}"
                ):
                    if agent['name'] not in st.session_state.selected_agents:
                        st.session_state.selected_agents.append(agent['name'])
                else:
                    if agent['name'] in st.session_state.selected_agents:
                        st.session_state.selected_agents.remove(agent['name'])
        
        st.divider()
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()
        
        st.divider()
        
        # System status
        st.subheader("System Status")
        try:
            status_response = requests.get(f"{API_BASE_URL}/status", timeout=5)
            if status_response.status_code == 200:
                status = status_response.json()
                st.success(f"Status: {status.get('status', 'unknown')}")
                st.text(f"Uptime: {status.get('uptime', 0):.0f}s")
            else:
                st.warning("API not available")
        except:
            st.error("Cannot connect to API")
        
        st.divider()
        
        # About
        st.markdown("""
        ### About
        
        Multi-Agent AI System combines specialized 
        AI agents to handle complex tasks.
        
        **Agents:**
        - 🔍 Research: Information retrieval
        - 📊 Analyst: Data analysis
        - 💻 Code: Code generation
        """)


def render_chat_message(role: str, content: str, metadata: Dict = None):
    """Render a chat message."""
    with st.chat_message(role):
        st.markdown(content)
        
        if metadata:
            with st.expander("Details"):
                if "execution_time" in metadata:
                    st.text(f"⏱️ Execution time: {metadata['execution_time']:.2f}s")
                if "agents_used" in metadata:
                    st.text(f"🤖 Agents: {', '.join(metadata['agents_used'])}")
                if "confidence" in metadata:
                    st.progress(metadata["confidence"], f"Confidence: {metadata['confidence']:.0%}")


def render_agent_trace(traces: list):
    """Render agent execution traces."""
    if not traces:
        return
    
    with st.expander("🔍 Agent Traces"):
        for trace in traces:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.text(f"Agent: {trace['agent']}")
            with col2:
                st.text(f"Duration: {trace['duration']:.2f}s")
            with col3:
                status_color = "🟢" if trace['status'] == "completed" else "🔴"
                st.text(f"{status_color} {trace['status']}")


def main():
    """Main application."""
    st.set_page_config(
        page_title="Multi-Agent AI System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.title("🤖 Multi-Agent AI System")
    st.markdown("*Enterprise-grade multi-agent orchestration with RAG capabilities*")
    
    # Display chat history
    for message in st.session_state.messages:
        render_chat_message(
            message["role"],
            message["content"],
            message.get("metadata")
        )
        if message.get("traces"):
            render_agent_trace(message["traces"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        render_chat_message("user", prompt)
        
        # Execute query
        with st.spinner("🔄 Processing..."):
            start_time = time.time()
            
            # Use selected agents or auto-select
            agents = st.session_state.selected_agents if st.session_state.selected_agents else None
            
            result = execute_query(prompt, agents)
            
            execution_time = time.time() - start_time
        
        # Process result
        if "error" in result:
            response_content = f"❌ Error: {result['error']}"
            metadata = None
            traces = None
        else:
            # Extract response
            response_content = result.get("result", {}).get("summary", "No response generated.")
            
            metadata = {
                "execution_time": result.get("execution_time", execution_time),
                "agents_used": [t["agent"] for t in result.get("agent_traces", [])],
                "confidence": result.get("result", {}).get("confidence", 0)
            }
            
            traces = result.get("agent_traces", [])
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_content,
            "metadata": metadata,
            "traces": traces
        })
        
        # Rerun to display
        st.rerun()
    
    # Quick actions
    st.divider()
    st.subheader("Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Analyze Data", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Analyze the latest data and provide key insights."
            })
            st.rerun()
    
    with col2:
        if st.button("🔍 Research", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Research the latest trends in AI and machine learning."
            })
            st.rerun()
    
    with col3:
        if st.button("💻 Generate Code", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Generate a Python function to process JSON data."
            })
            st.rerun()
    
    with col4:
        if st.button("📝 Summarize", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Summarize the key points from our knowledge base."
            })
            st.rerun()


if __name__ == "__main__":
    main()
