import streamlit as st
import sys
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from huggingface_hub import login
import gc

# Must be first Streamlit command
st.set_page_config(page_title="Energy Assistant", page_icon="⚡")

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def initialize_model():
    """Initialize model with memory-efficient settings"""
    try:
        # Clear memory first
        clear_memory()
        
        token = st.secrets["HUGGING_FACE_TOKEN"]
        login(token=token)
        st.success("✅ Authenticated with Hugging Face")

        # Load tokenizer first
        st.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2b",
            token=token,
            trust_remote_code=True,
            use_fast=True  # Use faster tokenizer
        )
        tokenizer.pad_token = tokenizer.eos_token
        st.success("✅ Tokenizer loaded")

        # Configure model loading for memory efficiency
        model_kwargs = {
            "token": token,
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "load_in_4bit": True,  # Use 4-bit quantization
            "device_map": "auto"
        }

        # Load base model
        st.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b",
            **model_kwargs
        )
        st.success("✅ Base model loaded")

        clear_memory()  # Clear memory before loading PEFT

        # Load PEFT model with minimal memory usage
        st.info("Loading PEFT model...")
        model = PeftModel.from_pretrained(
            base_model,
            "adetunjijeremiah/energy-gemma-2b",
            token=token,
            device_map="auto"
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
    st.write(f"Python: {sys.version}")
    st.write(f"PyTorch: {torch.__version__}")
    st.write(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Initialize session state
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
                    padding=True,
                    return_tensors="pt"
                ).to(st.session_state.model.device)

                outputs = st.session_state.model.generate(
                    **inputs,
                    max_length=256,  # Reduced for memory efficiency
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
                clear_memory()

if __name__ == "__main__":
    main()