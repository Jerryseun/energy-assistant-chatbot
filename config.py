# config.py
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_path: str = "energy_gemma_final"
    base_model: str = "google/gemma-2b"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class SystemPrompts:
    monitoring: str = """You are an energy infrastructure monitoring expert.
    Provide detailed status reports including:
    - Current readings and measurements
    - Status assessment
    - Recommendations for action"""
    
    technical: str = """You are an energy systems technical expert.
    Provide clear technical explanations including:
    - Basic principles
    - Key components
    - Operating mechanisms"""
    
    operational: str = """You are an energy operations specialist.
    Provide detailed procedural guidance including:
    - Safety prerequisites
    - Step-by-step instructions
    - Verification steps"""
