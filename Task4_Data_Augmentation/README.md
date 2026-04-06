This project is part of the Gen AI Internship at Elevance Skills.

# Task 4: Data Augmentation for Robust Colorization Models

## Problem Statement
A major challenge in training Generative AI for colorization is the lack of diverse training data. Models often overfit to specific textures or simple lighting conditions. This task focuses on developing a Robust Data Augmentation Pipeline that generates varied training samples from a base dataset, ensuring the colorization model can handle diverse orientations, noise levels, and structural variations found in real-world grayscale imagery.

## Dataset
The project utilized the COCO Val 2017 dataset as the primary source for high-quality RGB ground truth images. The augmentation pipeline transformed these images into a robust synthetic dataset of over 10,000 training pairs by injecting varied noise levels, resolution scaling, and geometric deformations to simulate a wide range of input qualities.

## Methodology
We implement a systematic augmentation workflow using torchvision and OpenCV. This includes synthetic grayscale conversion strategies to simulate different historic film types, geometric augmentations such as random flips and perspective shifts for structural invariance, and noise injection (Gaussian and Salt-and-Pepper) to simulate grainy historical film. Additionally, color jittering is applied to the target side during training to help the model learn a wider gamut of color possibilities and reduce color bleeding artifacts.

## Results
The augmentation pipeline significantly increases the effective size and diversity of the training dataset. By exposing the model to noisy and structurally varied versions of the data, we observed a measurable reduction in color bleeding and improved performance on low-quality archival scans. The resulting tensors are optimized for injection into a Stable Diffusion fine-tuning loop, providing a more generalized and robust colorization engine.

