import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load the model and tokenizer
config = PeftConfig.from_pretrained("adetunjijeremiah/energy-gemma-2b")
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
model = PeftModel.from_pretrained(base_model, "adetunjijeremiah/energy-gemma-2b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

# Function to interact with the model
def ask_model(question):
    inputs = tokenizer.encode(question, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("Energy Infrastructure Assistant")
st.write("Ask any question about energy infrastructure, and I will answer!")

# Text input from user
user_question = st.text_input("Your Question:", "")

# Generate and display the model's response
if user_question:
    response = ask_model(user_question)
    st.write("Assistant's Response:")
    st.write(response)
