import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import time

# Page config
st.set_page_config(page_title="Energy Assistant", page_icon="⚡")

# Initialize Hugging Face auth
if "HUGGING_FACE_TOKEN" in st.secrets:
    login(token=st.secrets["HUGGING_FACE_TOKEN"])

# Model loading with caching
@st.cache_resource
def load_model():
    try:
        # Load base tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2b",
            token=st.secrets["HUGGING_FACE_TOKEN"],
            trust_remote_code=True
        )
        
        # Load fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(
            "adetunjijeremiah/energy-gemma-2b",
            token=st.secrets["HUGGING_FACE_TOKEN"],
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Main app
st.title("⚡ Energy Infrastructure Assistant")

# Load model
if not st.session_state.model_loaded:
    try:
        with st.spinner("Loading model..."):
            model, tokenizer = load_model()
            st.session_state.model_loaded = True
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask about energy infrastructure..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Prepare input
                input_text = f"You are an energy infrastructure expert. Query: {prompt}\nResponse:"
                inputs = st.session_state.tokenizer(
                    input_text, 
                    return_tensors="pt"
                ).to(st.session_state.model.device)

                # Generate
                outputs = st.session_state.model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True
                )

                # Decode response
                response = st.session_state.tokenizer.decode(
                    outputs[0], 
                    skip_special_tokens=True
                ).replace(input_text, "").strip()

                st.write(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This AI assistant helps with:
    - 🔍 Equipment monitoring
    - 📚 Technical explanations
    - 🛠️ Operational procedures
    - 📋 Maintenance guidance
    """)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()