import streamlit as st
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Title and description
st.title("Energy Infrastructure Assistant")
st.write("Ask me anything about energy infrastructure!")

# Load PEFT-configured model and tokenizer
@st.cache_resource
def load_model():
    # Load the PEFT configuration
    config = PeftConfig.from_pretrained("adetunjijeremiah/energy-gemma-2b")
    
    # Load the base model and PEFT model
    base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
    model = PeftModel.from_pretrained(base_model, "adetunjijeremiah/energy-gemma-2b")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    
    return model, tokenizer

# Load the model and tokenizer
model, tokenizer = load_model()

# User input
user_input = st.text_area("Enter your query:")

# Generate response
if st.button("Submit"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            # Tokenize user input
            inputs = tokenizer.encode(user_input, return_tensors="pt")
            
            # Generate a response
            outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display the response
        st.write("### Response")
        st.write(response)
    else:
        st.warning("Please enter a query.")