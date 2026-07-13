import os


def maybe_load_lora(pipe, lora_path: str, lora_scale: float):
    """
    Loads LoRA weights if provided.
    Works when using diffusers pipelines that support load_lora_weights.
    """
    if not lora_path:
        return pipe

    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    # Diffusers supports LoRA loading in many pipelines
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora(lora_scale=lora_scale)
    return pipe
