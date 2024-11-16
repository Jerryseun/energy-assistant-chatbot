import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load your fine-tuned model and tokenizer from Hugging Face
config = PeftConfig.from_pretrained("adetunjijeremiah/energy-gemma-2b")
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
model = PeftModel.from_pretrained(base_model, "adetunjijeremiah/energy-gemma-2b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

# Function to interact with the model
def ask_model(question):
    inputs = tokenizer.encode(question, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio Interface
iface = gr.Interface(
    fn=ask_model,
    inputs=gr.Textbox(label="Your Question"),
    outputs=gr.Textbox(label="Assistant's Response"),
    title="Energy Infrastructure Assistant",
    description="Ask about energy infrastructure and get answers from the fine-tuned model."
)

# Launch the app
iface.launch()
