"""
Patches Task3 conditional_colorization.ipynb with the same speed optimizations from Task2.
Changes:
  1. Cell 'setup'  -> adds os env vars + safer cuda_available check
  2. Cell 'load_pipeline' -> adds VAE slicing/tiling, removes pipe.to(device) before offload, adds xformers try/except
  3. Cell 'gen_func' -> adds empty_cache, image resize to 512x512, reduces steps 25->10, guidance_scale 8.0->7.5, adds neg_prompt
"""
import json, pathlib

NB = pathlib.Path(r"d:\Downloads\image_gen_colorization\Task3_Conditional_Colorization\conditional_colorization.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    cid = cell.get("id", "")

    # ── Cell 1: setup ─────────────────────────────────────────────────────────
    if cid == "setup":
        cell["source"] = [
            "# Standard Imports\n",
            "import os\n",
            "os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'\n",
            "os.environ['CUDA_MODULE_LOADING'] = 'LAZY'\n",
            "\n",
            "import torch\n",
            "import numpy as np\n",
            "from PIL import Image\n",
            "import cv2\n",
            "from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler\n",
            "import gradio as gr\n",
            "\n",
            "# Device Setup\n",
            "cuda_available = torch.cuda.is_available()\n",
            "device = \"cuda\" if cuda_available else \"cpu\"\n",
            "dtype = torch.float16 if device == \"cuda\" else torch.float32\n",
            "print(f\"Using device: {device} with {dtype}\")",
        ]

    # ── Cell 2: load_pipeline ─────────────────────────────────────────────────
    elif cid == "load_pipeline":
        cell["source"] = [
            "print(\"Loading Conditional Colorization Models with 4GB VRAM Optimizations...\")\n",
            "controlnet = ControlNetModel.from_pretrained(\"lllyasviel/control_v11p_sd15_canny\", torch_dtype=dtype)\n",
            "pipe = StableDiffusionControlNetPipeline.from_pretrained(\n",
            "    \"runwayml/stable-diffusion-v1-5\", controlnet=controlnet, torch_dtype=dtype\n",
            ")\n",
            "pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)\n",
            "\n",
            "if device == \"cuda\":\n",
            "    # HIGH-SPEED 4GB OPTIMIZATIONS\n",
            "    pipe.enable_model_cpu_offload()  # Loads models one by one into GPU memory\n",
            "    pipe.enable_vae_slicing()         # Decodes image in slices to save VRAM\n",
            "    pipe.enable_vae_tiling()          # Allows processing of larger images in tiles\n",
            "    try:\n",
            "        pipe.enable_xformers_memory_efficient_attention()\n",
            "        print(\"xformers optimization: ENABLED\")\n",
            "    except Exception as e:\n",
            "        print(\"xformers not available, using attention slicing instead.\")\n",
            "        pipe.enable_attention_slicing()\n",
            "else:\n",
            "    pipe.to(device)\n",
            "\n",
            "print(\"Conditional Colorization Engine Ready!\")",
        ]

    # ── Cell 3: gen_func ──────────────────────────────────────────────────────
    elif cid == "gen_func":
        cell["source"] = [
            "def conditional_colorization(image, condition, custom_content=\"\"):\n",
            "    if torch.cuda.is_available():\n",
            "        torch.cuda.empty_cache()\n",
            "\n",
            "    # Resize to 512px for model efficiency & memory safety\n",
            "    image = image.convert(\"RGB\").resize((512, 512))\n",
            "\n",
            "    # Extract structural constraints\n",
            "    canny_map = extract_canny(image)\n",
            "\n",
            "    # Mapping Conditions to Atmospheric Prompts\n",
            "    condition_styles = {\n",
            "        \"Golden Hour (Sunset)\": \"warm golden lighting, orange and purple hues, sunset atmosphere, long shadows, soft sun glow\",\n",
            "        \"Cyberpunk (Neon)\": \"vibrant neon lights, cyan and magenta glow, futuristic night city atmosphere, rainy reflections\",\n",
            "        \"Overcast (Foggy)\": \"muted flat colors, soft diffused lighting, foggy atmosphere, gray and blue tones\",\n",
            "        \"Noon (Bright Sunlight)\": \"sharp direct sunlight, vibrant natural colors, deep shadows, high contrast daylight\",\n",
            "        \"Cinematic Teal & Orange\": \"cinematic color grading, teal shadows, orange highlights, film industry look\"\n",
            "    }\n",
            "\n",
            "    atmosphere_prompt = condition_styles.get(condition, \"realistic colors\")\n",
            "    full_prompt = f\"{custom_content}, in the style of {atmosphere_prompt}, high resolution, color-accurate\"\n",
            "    neg_prompt = \"blurry, low quality, distorted, unrealistic, cartoon, plastic\"\n",
            "\n",
            "    # Inference (10 steps for speed <20s on 4GB VRAM)\n",
            "    output = pipe(\n",
            "        full_prompt,\n",
            "        image=canny_map,\n",
            "        negative_prompt=neg_prompt,\n",
            "        num_inference_steps=10,\n",
            "        guidance_scale=7.5,\n",
            "        controlnet_conditioning_scale=1.0\n",
            "    ).images[0]\n",
            "\n",
            "    return output",
        ]

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("✅ Task3 notebook patched successfully!")
print("   - Added os env vars (LAZY CUDA loading)")
print("   - Added VAE slicing + VAE tiling")
print("   - Removed pipe.to(device) before cpu_offload")
print("   - Added VRAM empty_cache() before each generation")
print("   - Added image resize to 512x512")
print("   - Reduced inference steps: 25 → 10")
print("   - guidance_scale: 8.0 → 7.5")
print("   - Added negative_prompt")
