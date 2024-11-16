import streamlit as st

# Must be first Streamlit command
st.set_page_config(page_title="Energy Assistant", page_icon="⚡")

# Test imports
try:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    st.success("✅ Libraries imported successfully!")
except Exception as e:
    st.error(f"Error importing libraries: {str(e)}")
    st.stop()

# Display versions
st.write("Python version:", sys.version)
st.write("Torch version:", torch.__version__)
st.write("Transformers version:", transformers.__version__)

st.title("⚡ Energy Infrastructure Assistant")
st.write("Testing deployment...")