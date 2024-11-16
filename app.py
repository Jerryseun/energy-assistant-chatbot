import streamlit as st
import sys

# Must be first Streamlit command
st.set_page_config(page_title="Energy Assistant", page_icon="⚡")

st.title("⚡ Energy Infrastructure Assistant")

# Show system info
st.write("Testing environment...")
st.write("Python version:", sys.version)

# Test imports one by one
try:
    import torch
    st.write("PyTorch version:", torch.__version__)
except Exception as e:
    st.error(f"Failed to import torch: {str(e)}")

try:
    import transformers
    st.write("Transformers version:", transformers.__version__)
except Exception as e:
    st.error(f"Failed to import transformers: {str(e)}")

try:
    from peft import PeftModel, PeftConfig
    st.write("PEFT imported successfully")
except Exception as e:
    st.error(f"Failed to import PEFT: {str(e)}")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    st.write("Transformer models imported successfully")
except Exception as e:
    st.error(f"Failed to import transformer models: {str(e)}")

st.write("Environment test complete!")

# Basic interface placeholder
st.write("---")
st.write("Chat interface will be added once environment is verified.")