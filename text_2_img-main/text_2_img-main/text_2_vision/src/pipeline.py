import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline(model_id: str):
    device = get_device()

    # Autocast & fp16 on CUDA
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,  # for college demo; can be enabled if needed
    )

    # Better scheduler (often better quality)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)

    # Performance options
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

    return pipe, device
