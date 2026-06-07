import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
from src.config import NeuraCoderConfig
from src.model import NeuraCoder
from src.tokenizer import NeuraCoderTokenizer

app = FastAPI(title="NeuraCoder API", description="Code generation API for NeuraCoder")

# Global variables
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Code prompt")
    max_new_tokens: int = Field(200, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.1, le=2.0)
    top_k: int = Field(50, ge=0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)

class GenerateResponse(BaseModel):
    generated_code: str
    prompt_tokens: int
    generated_tokens: int

@app.on_event("startup")
def load_model():
    global model, tokenizer
    # مسیر مدل را می‌توان از متغیر محیطی یا پیش‌فرض استفاده کرد
    import os
    model_path = os.getenv("NEURACODER_MODEL_PATH", "checkpoints")
    tokenizer = NeuraCoderTokenizer()
    config = NeuraCoderConfig.from_pretrained(model_path)
    model = NeuraCoder(config).to(device)
    model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=device))
    model.eval()
    print(f"Model loaded from {model_path} on {device}")

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    input_ids = tokenizer.encode(request.prompt, add_special_tokens=True)
    prompt_tokens = len(input_ids)
    input_tensor = torch.tensor([input_ids], device=device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )
    
    generated = output_ids[0].tolist()
    generated_code = tokenizer.decode(generated[prompt_tokens:], skip_special_tokens=True)
    return GenerateResponse(
        generated_code=generated_code,
        prompt_tokens=prompt_tokens,
        generated_tokens=len(generated) - prompt_tokens
    )

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)