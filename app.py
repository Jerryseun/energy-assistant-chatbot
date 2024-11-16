import streamlit as st
st.set_page_config(page_title="Energy Assistant", page_icon="⚡")

# First, test imports with error handling
try:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    from huggingface_hub import login
    st.success("✅ All required packages imported successfully")
except Exception as e:
    st.error(f"Failed to import required packages: {str(e)}")
    st.stop()

def load_model():
    """Load the model with detailed error reporting"""
    try:
        # 1. Login to Hugging Face
        if "HUGGING_FACE_TOKEN" not in st.secrets:
            st.error("Missing Hugging Face token!")
            st.stop()
            
        token = st.secrets["HUGGING_FACE_TOKEN"]
        login(token=token)
        st.success("✅ Authenticated with Hugging Face")
        
        # 2. Load base model first
        base_model_id = "google/gemma-2b"
        st.info(f"Loading base model: {base_model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            token=token,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        st.success("✅ Base model loaded")
        
        # 3. Load tokenizer
        st.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            token=token,
            trust_remote_code=True
        )
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
        st.error(f"Error loading model: {str(e)}")
        st.exception(e)  # This will show the full error trace
        return None, None

def main():
    st.title("⚡ Energy Infrastructure Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load model if not already loaded
    if "model" not in st.session_state:
        with st.spinner("Loading model... This may take a few minutes..."):
            model, tokenizer = load_model()
            if model is not None and tokenizer is not None:
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.success("✅ Model initialized successfully!")
            else:
                st.error("Failed to initialize model")
                st.stop()
    
    # Simple chat interface for testing
    if prompt := st.chat_input("Ask about energy infrastructure..."):
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            try:
                inputs = st.session_state.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True
                )
                
                outputs = st.session_state.model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.7,
                    num_return_sequences=1
                )
                
                response = st.session_state.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                
                st.write(response)
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()