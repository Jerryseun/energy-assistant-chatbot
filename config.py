import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    # Using explicit paths
    base_model: str = "google/gemma-2b-it"  # Changed to instruction-tuned version
    model_path: str = "adetunjijeremiah/energy-gemma-2b"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"