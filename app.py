import streamlit as st
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from huggingface_hub import login

# Must be first Streamlit command
st.set_page_config(page_title="Energy Assistant", page_icon="⚡")

st.title("⚡ Energy Infrastructure Assistant")

def test_environment():
    """Test and display environment information"""
    st.write("Environment Information:")
    st.write("Python version:", sys.version)
    st.write("PyTorch version:", torch.__version__)
    st.write("Transformers version:", transformers.__version__)
    st.write("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    return True

def initialize_model():
    """Initialize model with step-by-step verification"""
    try:
        # 1. Verify Hugging Face token
        if "HUGGING_FACE_TOKEN" not in st.secrets:
            st.error("⚠️ Missing Hugging Face token!")
            return None, None
        
        token = st.secrets["HUGGING_FACE_TOKEN"]
        login(token=token)
        st.success("✅ Authenticated with Hugging Face")

        # 2. Load base model
        base_model_id = "google/gemma-2b"
        st.info(f"Loading base model: {base_model_id}")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            token=token,
            device_map="auto",
            torch_dtype=torch.float16
        )
        st.success("✅ Base model loaded")

        # 3. Load tokenizer
        st.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            token=token
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        st.success("✅ Tokenizer loaded")

        # 4. Load PEFT model
        peft_model_id = "adetunjijeremiah/energy-gemma-2b"
        st.info(f"Loading PEFT model: {peft_model_id}")
        
        model = PeftModel.from_pretrained(
            base_model,
            peft_model_id,
            token=token
        )
        st.success("✅ PEFT model loaded")

        return model, tokenizer

    except Exception as e:
        st.error(f"❌ Error initializing model: {str(e)}")
        st.exception(e)  # Show full error trace
        return None, None

def main():
    # Test environment first
    if test_environment():
        st.success("✅ Environment check passed")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load model if not already loaded
    if "model" not in st.session_state:
        with st.spinner("🔄 Loading model... Please wait..."):
            model, tokenizer = initialize_model()
            if model is not None and tokenizer is not None:
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.success("✅ Model initialized successfully!")
            else:
                st.error("❌ Failed to initialize model")
                st.stop()

    # Chat interface
    if "model" in st.session_state:
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
                    ).to(st.session_state.model.device)

                    outputs = st.session_state.model.generate(
                        **inputs,
                        max_length=512,
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
                    st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()