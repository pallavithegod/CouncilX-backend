import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Path to the fine-tuned model weights
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Reinforcement-learning/councilx_arbiter_final"))

class CouncilRewardModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Initializing Reward Model on {self.device} ---")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            self.model.eval()
            print("--- Reward Model Loaded Successfully ---")
        except Exception as e:
            print(f"--- Error Loading Reward Model: {e} ---")
            self.model = None

    def get_score(self, prompt: str, response: str) -> float:
        """
        Calculates a reward score based on log-likelihood. 
        Higher scores indicate the response aligns better with the fine-tuned 'Neutral' policy.
        """
        if not self.model:
            return 0.5
            
        full_text = f"User Query:\n{prompt}\n\nTask: Provide a neutral, objective final verdict.\n\nResponse:\n{response}"
        
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            
        # Convert loss (perplexity base) to a normalized 0-1 score
        # Lower loss = higher alignment = higher score
        import math
        perplexity = math.exp(loss) if loss < 10 else 1000
        score = max(0, min(1, 1 / (1 + (loss / 2)))) # Heuristic normalization
        
        return round(score, 3)

# Singleton instance
_reward_model_instance = None

def get_reward_score(prompt: str, response: str) -> float:
    global _reward_model_instance
    if _reward_model_instance is None:
        _reward_model_instance = CouncilRewardModel()
    return _reward_model_instance.get_score(prompt, response)
