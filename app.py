import streamlit as st
import sys
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from huggingface_hub import login

# Must be first Streamlit command
st.set_page_config(page_title="Energy Assistant", page_icon="⚡")

def initialize_model():
    """Initialize model with CPU settings"""
    try:
        token = st.secrets["HUGGING_FACE_TOKEN"]
        login(token=token)
        st.success("✅ Authenticated with Hugging Face")

        # Load tokenizer
        st.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2b",
            token=token,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        st.success("✅ Tokenizer loaded")

        # Load base model
        st.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b",
            token=token,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        st.success("✅ Base model loaded")

        # Load PEFT model
        st.info("Loading PEFT model...")
        model = PeftModel.from_pretrained(
            base_model,
            "adetunjijeremiah/energy-gemma-2b",
            token=token,
            device_map="cpu"
        )
        model.eval()
        st.success("✅ PEFT model loaded")

        return model, tokenizer

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
        return None, None

def main():
    st.title("⚡ Energy Infrastructure Assistant")
    
    # Show environment info
    st.write(f"Python version: {sys.version}")
    st.write(f"PyTorch version: {torch.__version__}")
    st.write(f"Transformers version: {transformers.__version__}")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load model
    if "model" not in st.session_state:
        with st.spinner("Loading model (this may take a few minutes)..."):
            model, tokenizer = initialize_model()
            if model and tokenizer:
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.success("✅ Model ready!")
            else:
                st.error("Failed to load model")
                st.stop()
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask about energy infrastructure..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            try:
                inputs = st.session_state.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    padding=True
                )

                outputs = st.session_state.model.generate(
                    **inputs,
                    max_length=256,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=st.session_state.tokenizer.pad_token_id
                )

                response = st.session_state.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                st.write(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()