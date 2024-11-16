import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import time
import streamlit as st
from config import ModelConfig, SystemPrompts
from huggingface_hub import login
import os

@st.cache_resource
def load_model(_model_path: str, _base_model: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer with caching"""
    try:
        # Login to Hugging Face
        if "HUGGING_FACE_TOKEN" in st.secrets:
            token = st.secrets["HUGGING_FACE_TOKEN"]
            login(token=token)
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            st.write("Successfully logged in to Hugging Face")
        else:
            st.error("HUGGING_FACE_TOKEN not found in secrets")
            st.stop()

        # Load tokenizer
        st.write(f"Loading tokenizer from {_base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(_base_model, trust_remote_code=True)
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model
        st.write(f"Loading model from {_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            _model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model.eval()
        st.write("Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

class EnergyBot:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.system_prompts = SystemPrompts()
        try:
            st.write(f"Initializing with base model: {config.base_model}")
            st.write(f"Loading fine-tuned model from: {config.model_path}")
            self.model, self.tokenizer = load_model(
                _model_path=config.model_path,
                _base_model=config.base_model
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
            st.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}", time.time() - start_time, 0