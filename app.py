import streamlit as st

# This must be the first Streamlit command
st.set_page_config(page_title="Energy Assistant Debug", page_icon="⚡")

# Now we can add other imports and debug info
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Debug information
st.title("Energy Assistant - Debug Mode")
st.write("Python version:", sys.version)
st.write("Current working directory:", os.getcwd())
st.write("Files in directory:", os.listdir())
st.write("PyTorch version:", torch.__version__)
st.write("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    st.write("CUDA device:", torch.cuda.get_device_name(0))

# Test Hugging Face token
if "HUGGING_FACE_TOKEN" in st.secrets:
    st.write("Hugging Face token found in secrets")
    try:
        login(token=st.secrets["HUGGING_FACE_TOKEN"])
        st.write("Successfully logged in to Hugging Face")
        
        # Test model loading
        with st.spinner("Testing model loading..."):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "google/gemma-2b",
                    token=st.secrets["HUGGING_FACE_TOKEN"],
                    trust_remote_code=True
                )
                st.success("✅ Tokenizer loaded successfully")
                
                model = AutoModelForCausalLM.from_pretrained(
                    "adetunjijeremiah/energy-gemma-2b",
                    token=st.secrets["HUGGING_FACE_TOKEN"],
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                st.success("✅ Model loaded successfully")
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                
    except Exception as e:
        st.error(f"Error logging in to Hugging Face: {str(e)}")
else:
    st.error("No Hugging Face token found in secrets")

# System info
st.subheader("System Information")
st.code(f"""
Python: {sys.version}
PyTorch: {torch.__version__}
CUDA: {torch.cuda.is_available()}
Directory: {os.getcwd()}
""")