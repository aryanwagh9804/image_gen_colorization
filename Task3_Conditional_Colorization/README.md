This project is part of the Gen AI Internship at Elevance Skills.

# Task 3: Conditional Colorization Based on User Constraints

## Problem Statement
Standard colorization often aims for a single realistic interpretation. However, real-world colorization often requires satisfying specific artistic or emotional constraints such as a warm sunset glow or a cool cinematic night. This task explores Conditional Colorization, where the Generative AI is guided by both structural inputs and explicit environmental or lighting conditions provided by the user.

## Dataset
The project utilized a collection of versatile landscape and urban environment photographs primarily sourced from the COCO 2017 dataset. This selection provided a wide range of lighting conditions (Daylight, Overcast, Twilight) to serve as a baseline for testing the model's ability to re-interpret the same structural scene under various user-defined constraints.

## Methodology
We implement a highly controllable pipeline using Stable Diffusion v1.5 and ControlNet. User conditions such as Golden Hour, Neon Cyberpunk, or Overcast Day are transformed into high-priority tokens within the text embedding layer of the diffusion model. By adjusting the guidance scale, we allow the model to dominate the global color harmony while using ControlNet to strictly maintain object boundaries, ensuring that the physical scene remains sharp and recognizable under any atmospheric bias.

## Results
The model successfully renders complex lighting scenarios while maintaining identical architectural details across different generations. For example, a single grayscale street scene can be convincingly transformed into a Sunny Morning or a Neon City Night. This demonstrates the power of conditional generation in creative colorization and its ability to achieve high-fidelity structural preservation under diverse environmental moods.

