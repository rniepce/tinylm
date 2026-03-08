"""
Unified Chat Server — All SLMs in one interface.

Serves:
  • MLX fine-tuned specialists (Literature, Jurídico, Data, Code)
  • tinyLM-pt (custom PyTorch transformer)

Usage:
    python serve.py [--port 8000]
"""
import argparse
import os
import sys
import yaml
import time
import json
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="SLM Hub")

# ── State ───────────────────────────────────────────────────────────────────
_state = {
    "model": None,
    "tokenizer": None,
    "current_model_name": None,
    "model_type": None,  # "mlx" or "tinylm"
    "model_info": {},
}


class ChatRequest(BaseModel):
    prompt: str
    system: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1


class SwitchRequest(BaseModel):
    model: str


# ── Model registry ──────────────────────────────────────────────────────────
MODELS = {
    "literature": {
        "display": "📚 English Literature",
        "description": "Literary analysis, creative writing, book discussion",
        "type": "mlx",
        "config": "models/literature/config.yaml",
    },
    "juridico": {
        "display": "⚖️ Jurídico TJMG",
        "description": "Direito processual civil, minutas, jurisprudência (PT-BR)",
        "type": "mlx",
        "config": "models/juridico/config.yaml",
    },
    "data-analyst": {
        "display": "📊 Data Analysis",
        "description": "SQL, pandas, statistics, data interpretation",
        "type": "mlx",
        "config": "models/data-analyst/config.yaml",
    },
    "code-assistant": {
        "display": "🧑‍💻 Code Assistant",
        "description": "Code generation, debugging, review, refactoring",
        "type": "mlx",
        "config": "models/code-assistant/config.yaml",
    },
    "tinylm": {
        "display": "🇧🇷 tinyLM-pt",
        "description": "30M param Portuguese transformer (treinado do zero)",
        "type": "tinylm",
        "path": "tinylm",
    },
}

SYSTEM_PROMPTS = {
    "literature": (
        "You are a literary scholar and creative writing expert specializing in English literature. "
        "You provide insightful analysis of literary works, discuss themes, symbolism, narrative "
        "techniques, and historical context. Respond thoughtfully and with academic depth."
    ),
    "juridico": (
        "Você é um assistente jurídico especializado em direito processual civil e na prática do "
        "Tribunal de Justiça de Minas Gerais (TJMG). Responda sempre em português brasileiro."
    ),
    "data-analyst": (
        "You are a senior data analyst expert in SQL, Python (pandas, numpy), "
        "statistics, and data visualization. You can work in both English and Portuguese."
    ),
    "code-assistant": (
        "You are an expert software engineer. You help with code generation, "
        "debugging, code review, and refactoring. Write clean, production-quality code."
    ),
    "tinylm": "",
}


# ── MLX loading ─────────────────────────────────────────────────────────────
def load_mlx_model(model_name: str):
    """Load an MLX specialist with LoRA adapter."""
    from mlx_lm import load

    info = MODELS[model_name]
    with open(info["config"]) as f:
        config = yaml.safe_load(f)

    base_model = config["model"]
    adapter_path = config.get("adapter_path", f"adapters/{model_name}")

    print(f"\n🔄 Loading MLX: {info['display']}")
    print(f"   Base: {base_model}")

    if os.path.exists(adapter_path) and os.listdir(adapter_path):
        print(f"   Adapter: {adapter_path}")
        model, tokenizer = load(base_model, adapter_path=adapter_path)
    else:
        print(f"   ⚠️  No adapter, loading base model")
        model, tokenizer = load(base_model)

    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["current_model_name"] = model_name
    _state["model_type"] = "mlx"
    _state["model_info"] = {
        "name": model_name,
        "display": info["display"],
        "description": info["description"],
        "base_model": base_model,
        "type": "mlx",
        "has_adapter": os.path.exists(adapter_path) and bool(os.listdir(adapter_path)),
    }
    print(f"   ✅ Ready!")


# ── tinyLM loading ──────────────────────────────────────────────────────────
def load_tinylm():
    """Load the custom PyTorch tinyLM model."""
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tinylm"))
    from tinylm.model import TinyLM
    from tinylm.config import Config
    from tokenizers import Tokenizer
    import glob

    tlm_dir = os.path.join(os.path.dirname(__file__), "tinylm")
    ckpt_pattern = os.path.join(tlm_dir, "checkpoints", "step_*.pt")
    files = sorted(glob.glob(ckpt_pattern))

    if not files:
        raise FileNotFoundError("No tinyLM checkpoint found")

    ckpt_path = files[-1]
    print(f"\n🔄 Loading tinyLM: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = TinyLM(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device(cfg.device)
    model = model.to(device)
    model.eval()

    tok_path = os.path.join(tlm_dir, cfg.tokenizer_path)
    tokenizer = Tokenizer.from_file(tok_path)

    params = model.count_parameters()
    print(f"   Params: {params/1e6:.1f}M │ Device: {cfg.device}")
    print(f"   ✅ Ready!")

    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["current_model_name"] = "tinylm"
    _state["model_type"] = "tinylm"
    _state["model_info"] = {
        "name": "tinylm",
        "display": "🇧🇷 tinyLM-pt",
        "description": "30M param Portuguese transformer (treinado do zero)",
        "type": "tinylm",
        "params": f"{params/1e6:.1f}M",
        "device": cfg.device,
    }
    # Store config and device for generation
    _state["tinylm_cfg"] = cfg
    _state["tinylm_device"] = device


# ── Routes ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    default = os.environ.get("SLM_MODEL", "literature")
    try:
        if MODELS[default]["type"] == "mlx":
            load_mlx_model(default)
        else:
            load_tinylm()
    except Exception as e:
        print(f"⚠️  Could not load '{default}': {e}")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/models")
async def list_models():
    result = {}
    for name, info in MODELS.items():
        adapter_exists = False
        if info["type"] == "mlx":
            adapter_path = f"adapters/{name}"
            adapter_exists = os.path.exists(adapter_path) and bool(
                os.listdir(adapter_path) if os.path.exists(adapter_path) else []
            )
        elif info["type"] == "tinylm":
            adapter_exists = os.path.exists("tinylm/checkpoints")

        result[name] = {
            **info,
            "has_adapter": adapter_exists,
            "is_active": name == _state["current_model_name"],
        }
    return result


@app.get("/api/info")
async def model_info():
    return _state["model_info"]


@app.post("/api/switch")
async def switch_model(req: SwitchRequest):
    if req.model == _state["current_model_name"]:
        return {"status": "already_loaded", **_state["model_info"]}

    if req.model not in MODELS:
        return {"status": "error", "message": f"Unknown model: {req.model}"}

    try:
        # Unload current model to free memory
        _state["model"] = None
        _state["tokenizer"] = None

        info = MODELS[req.model]
        if info["type"] == "mlx":
            load_mlx_model(req.model)
        else:
            load_tinylm()

        return {"status": "ok", **_state["model_info"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/generate")
async def generate(req: ChatRequest):
    if _state["model"] is None:
        return {"error": "No model loaded"}

    t0 = time.time()

    if _state["model_type"] == "mlx":
        response = generate_mlx(req)
    else:
        response = generate_tinylm(req)

    dt = time.time() - t0
    return {
        "text": response,
        "model": _state["current_model_name"],
        "time": round(dt, 2),
    }


def generate_mlx(req: ChatRequest):
    from mlx_lm import load, generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors

    model = _state["model"]
    tokenizer = _state["tokenizer"]
    model_name = _state["current_model_name"]

    system = req.system or SYSTEM_PROMPTS.get(model_name, "You are a helpful assistant.")

    # Some models (Mistral) don't support system role — prepend to user message
    # Try with system role first, fall back to prepending
    user_content = req.prompt
    try:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        messages = [
            {"role": "user", "content": f"{system}\n\n{user_content}"},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    sampler = make_sampler(temp=req.temperature, top_p=req.top_p)
    logits_processors = make_logits_processors(
        repetition_penalty=req.repetition_penalty
    )

    return mlx_generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=req.max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False,
    )


def generate_tinylm(req: ChatRequest):
    import torch

    model = _state["model"]
    tokenizer = _state["tokenizer"]
    device = _state["tinylm_device"]

    bos_id = tokenizer.token_to_id("<bos>")
    encoded = tokenizer.encode(req.prompt)
    tokens = [bos_id] + encoded.ids
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    output_ids = model.generate(
        idx,
        max_new_tokens=min(req.max_tokens, 300),
        temperature=req.temperature,
        top_k=50,
        top_p=req.top_p,
    )

    all_ids = output_ids[0].tolist()
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id in all_ids[1:]:
        all_ids = all_ids[1:all_ids.index(eos_id, 1)]
    else:
        all_ids = all_ids[1:]

    text = tokenizer.decode(all_ids)
    if text.startswith(req.prompt):
        text = text[len(req.prompt):]
    return text


# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="literature")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.environ["SLM_MODEL"] = args.model
    uvicorn.run(app, host="0.0.0.0", port=args.port)
