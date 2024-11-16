import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import time
import streamlit as st
from config import ModelConfig, SystemPrompts
from huggingface_hub import login

@st.cache_resource
def load_model(model_path: str, base_model: str):
    """Load model and tokenizer with caching"""
    try:
        # Login to Hugging Face
        if "HUGGING_FACE_TOKEN" in st.secrets:
            login(token=st.secrets["HUGGING_FACE_TOKEN"])
            print("Successfully logged in to Hugging Face")
        else:
            raise ValueError("HUGGING_FACE_TOKEN not found in secrets")

        print(f"Loading tokenizer from {base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            use_auth_token=st.secrets["HUGGING_FACE_TOKEN"]
        )
        
        print(f"Loading model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_auth_token=st.secrets["HUGGING_FACE_TOKEN"]
        )
        model.eval()
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

class EnergyBot:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.system_prompts = SystemPrompts()
        try:
            self.model, self.tokenizer = load_model(
                model_path=config.model_path,
                base_model=config.base_model
            )
        except Exception as e:
            st.error(f"Failed to initialize EnergyBot: {str(e)}")
            raise

    def _detect_query_type(self, query: str) -> str:
        """Detect query type to select appropriate system prompt"""
        query = query.lower()
        if any(word in query for word in ["status", "reading", "level", "current"]):
            return "monitoring"
        elif any(word in query for word in ["how", "explain", "what is", "describe"]):
            return "technical"
        elif any(word in query for word in ["procedure", "steps", "execute", "perform"]):
            return "operational"
        return "default"

    def generate_response(self, query: str) -> Tuple[str, float, int]:
        """Generate response with metrics"""
        start_time = time.time()
        try:
            # Get appropriate system prompt
            query_type = self._detect_query_type(query)
            system_prompt = getattr(self.system_prompts, query_type)
            
            # Format prompt
            prompt = f"{system_prompt}\n\nQuery: {query}\nResponse:"
            
            # Generate response
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
            
            # Process response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            token_count = len(self.tokenizer.encode(response))
            
            return response, elapsed_time, token_count
            
        except Exception as e:
            return f"Error generating response: {str(e)}", time.time() - start_time, 0