import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_path: str = "https://huggingface.co/adetunjijeremiah/energy-gemma-2b/tree/master"
    base_model: str = "google/gemma-2b"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class SystemPrompts:
    default: str = """You are an expert energy infrastructure assistant. 
    Provide clear, accurate responses about energy systems, monitoring, and operations."""
    
    monitoring: str = """You are monitoring energy infrastructure. 
    Provide detailed status reports including readings, status, and recommendations."""
    
    technical: str = """You are explaining technical aspects of energy systems. 
    Provide clear explanations of principles, components, and operations."""
    
    operational: str = """You are guiding energy operations. 
    Provide detailed procedures including safety steps and verifications."""