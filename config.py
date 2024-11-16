from dataclasses import dataclass

@dataclass
class ModelConfig:
    base_model: str = "gemma-2b-it"  # Changed to use model name only
    model_path: str = "adetunjijeremiah/energy-gemma-2b"  # Your fine-tuned model
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

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