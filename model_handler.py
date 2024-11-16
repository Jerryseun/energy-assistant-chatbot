import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import time
from config import ModelConfig, SystemPrompts
from huggingface_hub import login
import streamlit as st

class EnergyBot:
    def __init__(self, config: ModelConfig):
        self.config = config
        self._load_model()
        
    @st.cache_resource  # Cache the model loading to avoid reloading on every rerun
    def _load_model(self):
        """Initialize model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True
            )
            
            # Load the fine-tuned model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def generate_response(self, query: str) -> Tuple[str, float, int]:
        """Generate response with metrics"""
        start_time = time.time()
        try:
            # Format the prompt
            prompt = f"Query: {query}\nResponse:"
            
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode and clean response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            token_count = len(self.tokenizer.encode(response))
            
            return response, elapsed_time, token_count
            
        except Exception as e:
            return f"Error generating response: {str(e)}", time.time() - start_time, 0