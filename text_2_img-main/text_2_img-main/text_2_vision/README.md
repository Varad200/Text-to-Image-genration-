🎨 Text2Vision — Real-Time Text-to-Image Generator (GenAI)

Text2Vision is a lightweight, user-driven text-to-image generation system built around Stable Diffusion concepts.
The project provides a local, modular pipeline to generate images from text prompts, supports LoRA adapters and DreamBooth-style extensions, and includes utilities for prompt engineering, reproducibility, and output management.

🔗 Live Demo: https://text2vision-aryman.streamlit.app/

✨ Key Features

Real-time text-to-image generation using pretrained Stable Diffusion models

Prompt & negative prompt support for better control and quality

User-driven pipeline with configurable:

Guidance scale (CFG)

Seed (reproducibility)

Steps and resolution

Support for LoRA adapter weights (lightweight fine-tuning)

Scaffolded DreamBooth-style workflow for subject-specific extensions

Organized saving of generated images and metadata (JSON)

🧠 Core Concepts Used

Generative AI (Diffusion Models)

Prompt Engineering & Negative Prompting

Stable Diffusion (Pretrained Models)

GPU Acceleration (CUDA, optional)

Mixed Precision / Autocasting

Modular ML Pipeline Design

🛠 Requirements

Python 3.10+ (recommended)

GPU with CUDA for best performance (optional)

CPU-only execution is supported (slower inference)

All Python dependencies are listed in requirements.txt

🚀 Quick Start
1️⃣ Create & activate a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

2️⃣ Run the application
streamlit run app.py

3️⃣ Outputs

Generated images → output/images/

Generation metadata (JSON) → output/metadata/

🧩 Prompt Engineering

Recommended prompt structure:

subject + style + composition + lighting + quality details


Example:

cinematic portrait of a warrior, rim lighting, ultra detailed, 35mm lens


Negative prompts are used to remove unwanted artifacts such as blur, distortion, watermarks, or extra limbs.

Prompt templates and helpers are available in src/prompt_engineering.py.

🔁 Generation Pipeline (Workflow)

The complete image generation pipeline follows these steps:

User Input

User provides a text prompt and optional negative prompt

User selects parameters such as steps, guidance scale, seed, and resolution

Prompt Engineering

Prompts are processed using predefined templates

Enhancements improve style consistency and visual quality

Pipeline Initialization

A pretrained Stable Diffusion pipeline is loaded

CPU or GPU is selected automatically

Mixed precision (autocast) is enabled when GPU is available

Optional Model Extensions

LoRA adapters can be loaded to modify model behavior

DreamBooth-style extensions can be integrated for subject-specific tuning

Image Generation

The diffusion model iteratively denoises latent noise

Guidance scale controls how strongly the prompt influences the output

Output Handling

Generated images are saved to disk

Metadata (prompt, seed, parameters) is stored in JSON format

Visualization

Images are displayed in the web interface

Optional Matplotlib visualization is available for analysis

📁 Project Structure
text_to_image_GenAI/
│
├── app.py                  # Streamlit web application entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── src/
│   ├── pipeline.py         # Stable Diffusion pipeline loader & scheduler
│   ├── prompt_engineering.py  # Prompt templates and helpers
│   ├── lora.py             # LoRA loading and management utilities
│   └── utils.py            # Helper functions (saving outputs, metadata)
│
└── output/
    ├── images/             # Generated images
    └── metadata/           # JSON metadata for reproducibility

🔧 LoRA & DreamBooth Support

LoRA adapters allow fine-grained control of style or subject without retraining the full model.

DreamBooth-style workflows are scaffolded for future subject-specific fine-tuning.

These components are optional and intended for advanced experimentation.

⚡ Performance Tips

Use CUDA-enabled PyTorch for faster inference

Enable mixed precision (autocast) on GPUs

Prefer safe resolutions:

512 × 512

640 × 640

768 × 768

Reduce inference steps when running on CPU

🧪 Development Notes

Modify logic inside src/ and rerun streamlit run app.py

The project is designed with modularity and extensibility in mind

Suitable for college projects, internships, and GenAI demonstrations

🤝 Contributing

Contributions, issues, and suggestions are welcome.
Please include clear reproduction steps and environment details when reporting issues.

📜 License

This repository does not currently include a license file.
Add one if you plan to distribute or reuse the project commercially.

👤 Author & Contact

 Varad Bedre
📧 Email: varadbedre11@gmail.com