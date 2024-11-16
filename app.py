# Must be the first import and command
import streamlit as st
st.set_page_config(page_title="Energy Assistant", page_icon="⚡")

# Other imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

def initialize_model():
    """Initialize the model with error handling"""
    if "model" not in st.session_state:
        with st.spinner("Loading model... Please wait..."):
            try:
                # Login to Hugging Face
                if "HUGGING_FACE_TOKEN" not in st.secrets:
                    st.error("Missing Hugging Face token!")
                    st.stop()
                
                login(token=st.secrets["HUGGING_FACE_TOKEN"])
                
                # Load tokenizer
                st.info("Loading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    "google/gemma-2b",
                    use_auth_token=True,
                    trust_remote_code=True
                )
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                st.success("Tokenizer loaded!")
                
                # Load model with specific configuration
                st.info("Loading model...")
                model = AutoModelForCausalLM.from_pretrained(
                    "adetunjijeremiah/energy-gemma-2b/tree/master",
                    device_map="auto",
                    torch_dtype=torch.float16,
                    use_auth_token=True,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.success("Model loaded successfully!")
                
            except Exception as e:
                st.error(f"Error initializing model: {str(e)}")
                st.exception(e)  # This will show the full traceback
                st.stop()

def main():
    st.title("Energy Infrastructure Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize model
    initialize_model()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    if prompt := st.chat_input("Ask about energy infrastructure..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate response
        with st.chat_message("assistant"):
            try:
                full_prompt = f"Query: {prompt}\nResponse:"
                inputs = st.session_state.tokenizer(
                    full_prompt, 
                    return_tensors="pt"
                ).to(st.session_state.model.device)
                
                outputs = st.session_state.model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
                
                response = st.session_state.tokenizer.decode(
                    outputs[0], 
                    skip_special_tokens=True
                ).replace(full_prompt, "").strip()
                
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.exception(e)  # Show full traceback

if __name__ == "__main__":
    main()