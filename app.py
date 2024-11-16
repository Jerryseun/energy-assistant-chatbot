import streamlit as st
from model_handler import EnergyBot
from metrics_handler import ChatMetrics
from config import ModelConfig
import time
import traceback

# Page configuration
st.set_page_config(
    page_title="Energy Infrastructure Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize session state variables"""
    try:
        if 'model' not in st.session_state:
            with st.spinner('Loading model... This might take a minute...'):
                # Verify secrets
                if "HUGGING_FACE_TOKEN" not in st.secrets:
                    st.error("HUGGING_FACE_TOKEN not found in secrets!")
                    st.stop()
                
                config = ModelConfig()
                st.session_state.model = EnergyBot(config)
                st.success("Model loaded successfully!")
                
        if 'metrics' not in st.session_state:
            st.session_state.metrics = ChatMetrics()
        if 'messages' not in st.session_state:
            st.session_state.messages = []
    except Exception as e:
        st.error(f"Error initializing app: {str(e)}")
        st.error("Detailed error: " + traceback.format_exc())
        st.stop()


# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'model' not in st.session_state:
        config = ModelConfig()
        st.session_state.model = EnergyBot(config)
    if 'metrics' not in st.session_state:
        st.session_state.metrics = ChatMetrics()
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def render_sidebar():
    """Render sidebar with controls and documentation"""
    with st.sidebar:
        st.title("⚡ Energy Assistant")
        st.image("assets/logo.jpg", caption="Energy Assistant")
        
        st.markdown("""
        ### Features
        - 🔍 Real-time monitoring
        - 📊 Technical analysis
        - 🛠️ Operational procedures
        - 📋 Maintenance guidance
        """)
        
        # Settings and controls
        with st.expander("⚙️ Settings", expanded=False):
            st.session_state.max_length = st.slider(
                "Max Response Length", 
                100, 1000, 500
            )
            st.session_state.temperature = st.slider(
                "Temperature", 
                0.1, 1.0, 0.7
            )

def render_metrics():
    """Render metrics dashboard"""
    metrics = st.session_state.metrics
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Queries", metrics.total_queries)
    with col2:
        st.metric("Success Rate", f"{metrics.success_rate:.1f}%")
    with col3:
        st.metric("Avg Response Time", f"{metrics.average_response_time:.2f}s")
    with col4:
        st.metric("Avg Tokens", f"{metrics.average_token_count:.0f}")
    
    # Display metrics chart
    st.plotly_chart(metrics.create_metrics_chart(), use_container_width=True)

def main():
    # Initialize session
    init_session_state()
    
    # Page header
    st.title("Energy Infrastructure Assistant")
    st.markdown("Powered by Gemma-2B | Fine-tuned for energy domain expertise")
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if query := st.chat_input("Ask about energy infrastructure..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    response, response_time, token_count = st.session_state.model.generate_response(query)
                    st.markdown(response)
                    
                    # Update metrics
                    st.session_state.metrics.add_interaction(
                        response_time=response_time,
                        token_count=token_count,
                        success="Error" not in response
                    )
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        # Metrics dashboard
        render_metrics()

if __name__ == "__main__":
    main()