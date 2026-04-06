This project is part of the Gen AI Internship at Elevance Skills.

# Task 5: Semantic Segmentation for Region-Specific Colorization

## Problem Statement
A common failure mode in AI colorization is color bleeding, where colors from one object (for example, green from grass) leak into another (such as a person's skin). This occurs because the model lacks a high-level understanding of object boundaries. This task addresses this issue by integrating Semantic Segmentation into the colorization pipeline, ensuring that each region of the image is colorized according to its specific class and that object boundaries are strictly respected.

## Dataset
The project utilized the VOC and COCO datasets for training and evaluating the semantic segmentation backbone. For colorization testing, a subset of the COCO 2017 validation set was selected, specifically focusing on images with high object density and complex foreground-background relationships to rigorously test the model's ability to maintain distinct color boundaries.

## Methodology
We employ a two-stage AI pipeline involving semantic parsing and region-aware colorization. In the first stage, the input image is processed using a segmentation model (DeepLabV3) to identify regions such as sky, vegetation, human, and architecture. In the second stage, these segmentation masks are used to guide the diffusion process, either through masked refinement in the attention layers or by providing localized color priors. This ensures that the generated colors stay strictly within their semantic boundaries and prevents unrealistic color overlaps.

## Results
The segmented approach yields significantly higher perceptual quality scores compared to global colorization methods. Colors are naturally distributed according to object classes, and color bleeding artifacts are effectively eliminated. The integration of semantic masks allows for fine-grained detail preservation, ensuring sharp transitions between high-contrast objects and a more realistic overall appearance.

