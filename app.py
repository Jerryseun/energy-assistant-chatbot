import streamlit as st
from model_handler import EnergyBot
from config import ModelConfig
import time

# Page configuration
st.set_page_config(
    page_title="Energy Infrastructure Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'model' not in st.session_state:
        try:
            with st.spinner('Loading model... This might take a minute...'):
                config = ModelConfig()
                st.session_state.model = EnergyBot(config)
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            st.stop()

def main():
    st.title("⚡ Energy Infrastructure Assistant")
    
    # Initialize session
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        This assistant helps with:
        - 🔍 Equipment monitoring
        - 📚 Technical explanations
        - 🛠️ Operational procedures
        - 📋 Maintenance guidance
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about energy infrastructure..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, response_time, token_count = st.session_state.model.generate_response(prompt)
                st.write(response)
                
                # Show metrics in expander
                with st.expander("Response Metrics"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Response Time", f"{response_time:.2f}s")
                    with col2:
                        st.metric("Token Count", token_count)
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()