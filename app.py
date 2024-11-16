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

                # Load tokenizer
                st.info("📚 Loading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    "google/gemma-2b",
                    token=token,  # Use token instead of use_auth_token
                    trust_remote_code=True
                )
                tokenizer.pad_token = tokenizer.eos_token
                st.success("✅ Tokenizer loaded")

                # Load model with specific Gemma configurations
                st.info("🔄 Loading model (this may take a few minutes)...")
                model = AutoModelForCausalLM.from_pretrained(
                    "adetunjijeremiah/energy-gemma-2b",
                    token=token,  # Use token instead of use_auth_token
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    config_overrides={
                        "hidden_activation": "gelu_pytorch_tanh"
                    }
                )
                
                # Tie weights explicitly
                model.tie_weights()
                
                # Move model to device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.success("✅ Model loaded successfully!")
                st.info(f"💻 Using device: {device}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()

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
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                with st.spinner("🤔 Thinking..."):
                    # Prepare input
                    input_text = f"Query: {prompt}\nResponse:"
                    inputs = st.session_state.tokenizer(
                        input_text,
                        return_tensors="pt",
                        padding=True
                    ).to(st.session_state.model.device)

                    # Generate
                    outputs = st.session_state.model.generate(
                        **inputs,
                        max_length=512,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=st.session_state.tokenizer.pad_token_id,
                        eos_token_id=st.session_state.tokenizer.eos_token_id
                    )

                    # Decode response
                    response = st.session_state.tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True
                    ).replace(input_text, "").strip()

                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")

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