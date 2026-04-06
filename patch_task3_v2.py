import json
import os

notebook_path = r'd:\Downloads\image_gen_colorization\Task3_Conditional_Colorization\conditional_colorization.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell that starts with "clear_memory()" and "Loading Conditional Pipeline..."
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Loading Conditional Pipeline' in "".join(cell['source']):
        cell['source'] = [
            "clear_memory()\n",
            "print(\"Loading Conditional Pipeline with 4GB VRAM High-Speed Optimizations...\")\n",
            "controlnet = ControlNetModel.from_pretrained(\"lllyasviel/control_v11p_sd15_canny\", torch_dtype=dtype)\n",
            "pipe = StableDiffusionControlNetPipeline.from_pretrained(\n",
            "    \"runwayml/stable-diffusion-v1-5\", controlnet=controlnet, torch_dtype=dtype\n",
            ")\n",
            "pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)\n",
            "\n",
            "if device == \"cuda\":\n",
            "    # HIGH-SPEED 4GB OPTIMIZATIONS (Matches Task 2)\n",
            "    pipe.enable_model_cpu_offload() # Crucial for 4GB RTX 3050 to avoid slow shared memory\n",
            "    pipe.enable_vae_slicing()\n",
            "    pipe.enable_vae_tiling()\n",
            "    try:\n",
            "        pipe.enable_xformers_memory_efficient_attention()\n",
            "        print(\"xformers optimization: ENABLED\")\n",
            "    except:\n",
            "        pipe.enable_attention_slicing(\"max\")\n",
            "        print(\"xformers not available, using max attention slicing.\")\n",
            "else:\n",
            "    pipe.to(device)\n",
            "\n",
            "print(\"Conditional Colorization Engine Ready (Optimized for RTX 3050)!\")"
        ]
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Task 3 Notebook Optimized Successfully!")
