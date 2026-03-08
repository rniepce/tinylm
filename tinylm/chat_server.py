"""
Chat API server for tinyLM-pt.

Loads the trained model and exposes a streaming-compatible
generation endpoint for the chat UI.

Usage:
    pip install fastapi uvicorn
    python chat_server.py
"""
import os
import glob
import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from tokenizers import Tokenizer
from config import Config
from model import TinyLM

app = FastAPI(title="tinyLM-pt Chat")

# ── Global model state ──────────────────────────────────────────────────────
model = None
tokenizer = None
cfg = None


class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9


def load_model():
    global model, tokenizer, cfg

    # Find latest checkpoint
    pattern = os.path.join("checkpoints", "step_*.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No checkpoint found. Train the model first.")

    ckpt_path = files[-1]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = TinyLM(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device(cfg.device)
    model = model.to(device)
    model.eval()

    tokenizer = Tokenizer.from_file(cfg.tokenizer_path)

    step = ckpt.get("step", "?")
    loss = ckpt.get("loss", "?")
    params = model.count_parameters()
    print(f"✅ Model loaded: {ckpt_path}")
    print(f"   Step: {step} │ Loss: {loss:.6f}")
    print(f"   Device: {cfg.device} │ Params: {params/1e6:.1f}M")


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/info")
async def model_info():
    return {
        "params": f"{model.count_parameters()/1e6:.1f}M",
        "device": cfg.device,
        "vocab_size": cfg.vocab_size,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "max_seq_len": cfg.max_seq_len,
    }


@app.post("/api/generate")
async def generate(req: ChatRequest):
    device = next(model.parameters()).device
    bos_id = tokenizer.token_to_id("<bos>")
    encoded = tokenizer.encode(req.prompt)
    tokens = [bos_id] + encoded.ids
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    output_ids = model.generate(
        idx,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )

    all_ids = output_ids[0].tolist()
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id in all_ids[1:]:
        all_ids = all_ids[1:all_ids.index(eos_id, 1)]
    else:
        all_ids = all_ids[1:]

    text = tokenizer.decode(all_ids)
    # Remove the original prompt from the output
    if text.startswith(req.prompt):
        text = text[len(req.prompt):]

    return {"text": text, "prompt": req.prompt}


# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
