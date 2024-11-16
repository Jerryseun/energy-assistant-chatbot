import streamlit as st
st.set_page_config(page_title="Energy Assistant", page_icon="⚡")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import time

def initialize_model():
    """Initialize the model with robust error handling"""
    if "model" not in st.session_state:
        with st.spinner("🔄 Initializing..."):
            try:
                # Login to Hugging Face
                token = st.secrets["HUGGING_FACE_TOKEN"]
                login(token=token)
                st.info("🔑 Authenticated with Hugging Face")

                # Load base model and tokenizer
                base_model_id = "google/gemma-2b"
                st.info(f"📚 Loading model from {base_model_id}...")

                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_id,
                    token=token,
                    trust_remote_code=True
                )
                tokenizer.pad_token = tokenizer.eos_token
                st.success("✅ Tokenizer loaded")

                model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    token=token,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )

                # Verify model loaded correctly
                if model is None:
                    raise ValueError("Model failed to load")

                # Move model to device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                model.eval()  # Set to evaluation mode
                
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.success("✅ Model loaded successfully!")
                st.info(f"💻 Using device: {device}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()

def generate_response(prompt: str) -> str:
    """Generate response with energy domain prompt engineering"""
    system_prompt = """You are an expert energy infrastructure assistant. 
    Provide detailed and accurate responses about energy systems, monitoring, and operations.
    Focus on clear, actionable information and safety considerations."""
    
    full_prompt = f"{system_prompt}\n\nQuery: {prompt}\nResponse:"
    
    try:
        inputs = st.session_state.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True
        ).to(st.session_state.model.device)

        outputs = st.session_state.model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=st.session_state.tokenizer.pad_token_id,
            eos_token_id=st.session_state.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

        response = st.session_state.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).replace(full_prompt, "").strip()

        return response

    except Exception as e:
        st.error(f"❌ Error generating response: {str(e)}")
        return "I apologize, but I encountered an error generating the response. Please try again."

def main():
    st.title("⚡ Energy Infrastructure Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize model
    initialize_model()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about energy infrastructure..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = generate_response(prompt)
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

    # Sidebar
    with st.sidebar:
        st.markdown("""
        ### About
        This AI assistant specializes in:
        - 🔍 Equipment monitoring
        - 📚 Technical explanations
        - 🛠️ Operational procedures
        - 📋 Maintenance guidance
        """)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()