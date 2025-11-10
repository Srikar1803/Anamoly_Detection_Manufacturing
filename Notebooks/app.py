
import os, json, yaml, torch, numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import gradio as gr

# your project modules
from models.autoencoder import UNetAE
from losses import topk_error

RUNS = Path("runs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_one_category(run_dir: Path):
    """Load model, transform, threshold, topk for a single category directory."""
    cfg = yaml.safe_load(open(run_dir / "config_updated.yaml"))
    model = UNetAE(base=int(cfg["base_channels"]), latent=int(cfg["latent_dim"])).to(DEVICE)
    ckpt = torch.load(run_dir / "best.pth", map_location=DEVICE)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    tfm = T.Compose([T.Resize((int(cfg["image_size"]), int(cfg["image_size"]))), T.ToTensor()])
    thr = float(json.load(open(run_dir / "threshold.json"))["threshold"])
    topk = float(cfg.get("topk_frac", 0.01))
    return model, tfm, thr, topk

def load_all():
    """Scan runs/ for valid categories and cache models in memory."""
    cats = []
    models = {}
    if not RUNS.exists():
        return cats, models

    for d in sorted([p for p in RUNS.iterdir() if p.is_dir()]):
        if not (d / "best.pth").exists():           continue
        if not (d / "config_updated.yaml").exists(): continue
        if not (d / "threshold.json").exists():      continue
        try:
            models[d.name] = _load_one_category(d)
            cats.append(d.name)
        except Exception as e:
            print(f"Skip {d.name}: {e}")
    return cats, models

CATEGORIES, MODELS = load_all()

def infer_one(model, tfm, img_pil, topk_frac=0.01):
    """Return (score, heatmap) given a PIL image."""
    x = tfm(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y = model(x)
        s = topk_error(y, x, topk_frac=topk_frac)
        # topk_error may return tensor or tuple; handle both
        if hasattr(s, "__len__"):
            score = float(s[0].item())
        else:
            score = float(s.item())
        heat = (y - x).abs().mean(dim=1)[0].detach().cpu().numpy()
        # normalize heat for visualization
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
    return score, heat

def to_overlay(img_pil, heat):
    """RGB overlay image (numpy uint8) combining input and heatmap."""
    base = np.asarray(img_pil.convert("RGB")).astype(np.float32) / 255.0
    h = np.array(Image.fromarray((heat * 255).astype(np.uint8)).resize(
        (base.shape[1], base.shape[0]), Image.BILINEAR)) / 255.0
    # blue-red overlay (red for high anomaly)
    overlay = (0.6 * base + 0.4 * np.stack([h, h * 0.2, 1 - h], axis=-1)).clip(0, 1)
    return (overlay * 255).astype(np.uint8)

def predict(category, image):
    """Gradio interface fn."""
    if image is None:
        return "Upload an image to get a prediction.", None, None

    if category not in MODELS:
        return f"Category '{category}' not loaded/found under runs/.", None, None

    model, tfm, thr, topk = MODELS[category]

    # Gradio passes numpy array or PIL; ensure PIL
    if isinstance(image, np.ndarray):
        img_pil = Image.fromarray(image.astype(np.uint8))
    elif isinstance(image, Image.Image):
        img_pil = image
    else:
        return "Unsupported image type.", None, None

    score, heat = infer_one(model, tfm, img_pil, topk_frac=topk)
    pred = "ANOMALY" if score >= thr else "OK"

    overlay = to_overlay(img_pil, heat)

    header = f"{category}: {pred}\nscore={score:.6f} | threshold={thr:.6f}"
    return header, heat, overlay

with gr.Blocks(title="MVTec Anomaly Gradio") as demo:
    gr.Markdown("# Per-Category Anomaly Detection (Gradio)")

    with gr.Row():
        category = gr.Dropdown(
            CATEGORIES, value=CATEGORIES[0] if CATEGORIES else None, label="Category"
        )
        img = gr.Image(type="pil", label="Upload Image")

    btn = gr.Button("Predict")
    out_text = gr.Textbox(label="Prediction", lines=2)
    out_heat = gr.Image(label="Heatmap (normalized)", image_mode="L")
    out_overlay = gr.Image(label="Overlay")

    btn.click(predict, inputs=[category, img], outputs=[out_text, out_heat, out_overlay])

# For script usage:
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
