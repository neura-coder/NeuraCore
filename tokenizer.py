from transformers import AutoTokenizer
from typing import List

class NeuraCoderTokenizer:
    def __init__(self, model_name: str = "codellama/CodeLlama-7b-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        special_tokens = ["<code>", "</code>", "<python>", "<javascript>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.vocab_size = len(self.tokenizer)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def __call__(self, text, return_tensors="pt", **kwargs):
        return self.tokenizer(text, return_tensors=return_tensors, **kwargs)
    
    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id
    
    def save_pretrained(self, path):
        self.tokenizer.save_pretrained(path)