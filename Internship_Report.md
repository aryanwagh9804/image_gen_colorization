# Internship Final Report: Real-time Gen AI Image Colorization

## 1. Introduction
This report summarizes the work completed during the Gen AI Internship at **Elevance Skills**. The project focused on extending the original Stable Diffusion-based Colorization training into a commercial-grade, multi-functional research project. We successfully implemented six advanced features, each addressing a unique challenge in the field of AI-driven image restoration and creative colorization.

## 2. Problem Statement
Colorization is a complex 'inverse problem' in computer vision. Grayscale images lack the color information needed to reconstruct a 'true' reality. Traditional methods often produce flat, unrealistic colors or suffer from 'color bleeding'. The goal of this internship was to build a robust, integrated framework using **Generative AI (Stable Diffusion)** and **ControlNet** to solve these issues across diverse domains: from historical restoration to artistic styling and video processing.

## 3. Methodology
The project was built on **Stable Diffusion v1.5** with **ControlNet (Canny/Lineart)** as the structural backbone. The methodology involved:
- **Modular Task Design**: Each of the 6 internship tasks was developed as a standalone feature within an integrated repository.
- **Advanced Prompt Engineering**: Leveraged specialized tokens to guide the AI towards specific historical eras, artistic styles, and atmospheric conditions.
- **Multimodal Integration**: Combined Diffusion models with Semantic Segmentation (DeepLabV3) and Video Processing (OpenCV) to achieve localized and temporal accuracy.
- **Optimization**: Used `float16` precision, `xformers`, and `model CPU offloading` to ensure the pipeline runs efficiently on consumer-grade hardware.

## 4. Tasks Summary & Results

| Task # | Feature Name | Key Outcome |
| :--- | :--- | :--- |
| **Task 1** | Artistic Style Transfer | Successfully colorized images in the style of Van Gogh, Watercolor, and Cyberpunk. |
| **Task 2** | Historical Restoration | Achieved authentic colorization for WWII-era and 1920s archival photos. |
| **Task 3** | Conditional Colorization | Modeled atmospheric conditions like 'Golden Hour' and 'Neon Night' accurately. |
| **Task 4** | Data Augmentation | Built a robust pipeline for generating synthetic training pairs with noise injection. |
| **Task 5** | Semantic Segmentation | Integrated object-level masks to eliminate color bleeding across boundaries. |
| **Task 6** | Real-time Video & Live Feed | Optimized the diffusion pipeline for flicker-reduced video and live webcam colorization. |

## 5. Challenges Faced
- **Hardware Constraints**: Managing the VRAM requirements of Stable Diffusion while running Gradio and segmentation models simultaneously. Resolved using memory-efficient attention and CPU offloading.
- **Temporal Consistency**: Reducing flickering in Task 6 required fine-tuning the inference steps and guidance scale to find a balance between stability and detail.
- **Historical Accuracy**: Finding the right prompt triggers to avoid modern 'digital' looks in archival restoration. This was overcome through extensive research into 1940s film technology.

## 6. Conclusion & Outcomes
The internship project has resulted in a professional-grade Gen AI repository. We have moved beyond simple 'text-to-image' and created a specialized 'image-to-image' restoration engine. All code is original, well-documented, and reproducible. 

**Deliverables Included:**
- 6 Standalone Jupyter Notebooks.
- 6 Professional README documents for each task.
- Integrated `requirements.txt` for the entire ecosystem.
- This Final Internship Report.

---
**Internship Portfolio link**: [GitHub Repository - Real-time Gen AI Colorization](https://github.com/user/image_gen_colorization) (Public Link Placeholder)
**Daily Report Status**: Completed via [Daily Report Form](https://forms.gle/oi3UhE4RcZEKSu1r5).

*Submitted to: Elevance Skills Review Team*
*Date: April 2, 2026*
