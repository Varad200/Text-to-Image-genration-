import os
import torch
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from contextlib import nullcontext

from contextlib import nullcontext
import time


from src.pipeline import load_pipeline
from src.prompt_engineering import PROMPT_TEMPLATES, NEGATIVE_DEFAULT
from src.utils import ensure_dirs, save_metadata
from src.lora import maybe_load_lora

ensure_dirs()

st.set_page_config(page_title="Text-to-Image GenAI", layout="wide")
st.title("🖼️ Real-Time Text-to-Image Generator (Gen AI)")
st.caption("Stable Diffusion + Prompt Engineering + GPU/CPU Support (College Project)")

# ---------------- Sidebar: User-driven pipeline controls ----------------
st.sidebar.header("⚙️ Controls (User-driven pipeline)")

model_id = st.sidebar.selectbox(
    "Stable Diffusion model (pretrained)",
    [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1-base",
    ],
    index=0,
)

template = st.sidebar.selectbox(
    "Prompt template (Prompt Engineering)", list(PROMPT_TEMPLATES.keys()), index=0
)

steps = st.sidebar.slider("Steps (quality vs speed)", 10, 60, 30)
guidance = st.sidebar.slider(
    "Guidance Scale (CFG)", 1.0, 15.0, 7.5
)  # Guidance Scale Control
seed = st.sidebar.number_input(
    "Seed (reproducible output)", min_value=0, value=42, step=1
)

width = st.sidebar.selectbox("Width", [512, 640, 768], index=0)
height = st.sidebar.selectbox("Height", [512, 640, 768], index=0)

num_images = st.sidebar.slider("Images per prompt", 1, 4, 1)

st.sidebar.subheader("LoRA Fine-Tuning (optional)")
lora_path = st.sidebar.text_input("LoRA path (.safetensors)", value="")
lora_scale = st.sidebar.slider("LoRA scale", 0.1, 1.0, 0.7)

st.sidebar.subheader("Performance")
use_autocast = st.sidebar.checkbox("Use autocast (GPU mixed precision)", value=True)

# ---------------- Main UI ----------------
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_area(
        "Enter your prompt", value="a futuristic city skyline at sunset, ultra detailed"
    )
    negative_prompt = st.text_area("Negative prompt", value=NEGATIVE_DEFAULT)
    run = st.button("🚀 Generate")

with col2:
    st.markdown("### Prompt Tips (Prompt Engineering)")
    st.write("- Add style keywords: *cinematic, ultra detailed, 8k*")
    st.write("- Add composition: *wide shot, close-up, depth of field*")
    st.write("- Add lighting: *soft light, rim light, neon glow*")
    st.write("- Use negative prompts to remove artifacts.")

# ---------------- Load pipeline (cached in session) ----------------
if "pipe" not in st.session_state or st.session_state.get("model_id") != model_id:
    with st.spinner("Loading Stable Diffusion model..."):
        pipe, device = load_pipeline(model_id)
        st.session_state["pipe"] = pipe
        st.session_state["device"] = device
        st.session_state["model_id"] = model_id

pipe = st.session_state["pipe"]
device = st.session_state["device"]

# ---------------- Generation ----------------
if run:
    final_prompt = PROMPT_TEMPLATES[template].format(prompt=prompt)

    # Seed handling
    generator = torch.Generator(device=device).manual_seed(int(seed))

    # LoRA (if provided)
    try:
        if lora_path.strip():
            pipe = maybe_load_lora(pipe, lora_path.strip(), lora_scale)
    except Exception as e:
        st.error(f"LoRA load failed: {e}")

    with st.spinner("Generating image(s)..."):
        # Create placeholders for metrics
        metric_col1, metric_col2 = st.columns(2)
        progress_placeholder = st.empty()
        
        start_time = datetime.now()
        
        # Autocast improves CUDA performance
        autocast_ctx = (
            torch.autocast("cuda")
            if (device == "cuda" and use_autocast)
            else nullcontext()
        )


        with autocast_ctx:
            out = pipe(
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                width=int(width),
                height=int(height),
                num_images_per_prompt=int(num_images),
                generator=generator,
            )
        
        # Calculate metrics
        end_time = datetime.now()
        time_taken = (end_time - start_time).total_seconds()
        
        # Display metrics with live updates
        with metric_col1:
            status_placeholder = st.empty()
            status_placeholder.metric("Status", "Processing...", delta="⏳")
        with metric_col2:
            time_placeholder = st.empty()
            time_placeholder.metric("Time Taken", "0.00s")
        
        # Update metrics in real-time during generation
        start_tick = time.time()
        while True:
            elapsed = time.time() - start_tick
            time_placeholder.metric("Time Taken", f"{elapsed:.2f}s")
            if elapsed > time_taken:
                status_placeholder.metric("Status", "Complete", delta="✅ Done")
            break
            time.sleep(0.1)
        # Autocast improves CUDA performance
        autocast_ctx = (
            torch.autocast("cuda")
            if (device == "cuda" and use_autocast)
            else nullcontext()
        )

        with autocast_ctx:
            out = pipe(
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                width=int(width),
                height=int(height),
                num_images_per_prompt=int(num_images),
                generator=generator,
            )

    images = out.images

    st.success(f"✅ Generated {len(images)} image(s) on {device.upper()}")

    # Save outputs + metadata
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = []
    for i, img in enumerate(images):
        path = f"outputs/images/{ts}_{i}.png"
        img.save(path)
        saved_paths.append(path)

    meta_path = save_metadata(
        {
            "timestamp": ts,
            "model_id": model_id,
            "device": device,
            "prompt": final_prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "width": width,
            "height": height,
            "num_images": num_images,
            "lora_path": lora_path,
            "lora_scale": lora_scale,
            "saved_images": saved_paths,
        }
    )

    st.write("📄 Metadata saved to:", meta_path)

    # Display gallery
    st.markdown("## 🖼️ Output Gallery")
    cols = st.columns(min(4, len(images)))
    for idx, img in enumerate(images):
        with cols[idx % len(cols)]:
            st.image(img, use_container_width=True)
            st.caption(saved_paths[idx])

    # Matplotlib Visualization
    st.markdown("## 📊 Matplotlib Visualization")
    fig, ax = plt.subplots()
    ax.imshow(images[0])
    ax.axis("off")
    st.pyplot(fig)