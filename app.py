import streamlit as st
import sys
import os

# First, let's print debug information
st.write("Python version:", sys.version)
st.write("Current working directory:", os.getcwd())
st.write("Files in directory:", os.listdir())

# Try importing required packages one by one
try:
    import torch
    st.write("PyTorch version:", torch.__version__)
except Exception as e:
    st.error(f"Error importing torch: {str(e)}")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    st.write("Transformers imported successfully")
except Exception as e:
    st.error(f"Error importing transformers: {str(e)}")

try:
    from huggingface_hub import login
    st.write("Hugging Face Hub imported successfully")
except Exception as e:
    st.error(f"Error importing huggingface_hub: {str(e)}")

# Basic page config
st.set_page_config(page_title="Energy Assistant Debug", page_icon="⚡")
st.title("Energy Assistant - Debug Mode")

# Test Hugging Face token
if "HUGGING_FACE_TOKEN" in st.secrets:
    st.write("Hugging Face token found in secrets")
    try:
        login(token=st.secrets["HUGGING_FACE_TOKEN"])
        st.write("Successfully logged in to Hugging Face")
    except Exception as e:
        st.error(f"Error logging in to Hugging Face: {str(e)}")
else:
    st.error("No Hugging Face token found in secrets")

# Basic interface
st.write("Debug mode active - testing basic functionality")