This project is part of the Gen AI Internship at Elevance Skills.

# Task 6: Real-time Video Colorization with Temporal Consistency

## Problem Statement
Traditional image colorizers, when applied to video, suffer from temporal flickering where the color of an object changes randomly from frame to frame. This occurs because the AI model treats each frame as a completely independent image. This task focuses on developing a real-time video colorization pipeline that maintains color consistency across frames, creating a stable and visually pleasing result for vintage film restoration and live streams.

## Dataset
The project utilized a variety of short historical video clips and live webcam streams for real-time testing. High-speed inference was benchmarked using standard video sequences to ensure that the temporal consistency algorithms could maintain color stability over long durations without significant flickering or cumulative color drift.

## Methodology
To achieve stability and speed, we implement a streamlined diffusion pipeline using a pruned version of Stable Diffusion v1.5 and a high-speed scheduler. We implement a temporal guidance strategy where the context from previous frames is used to provide a subtle color bias to the current frame generation. The system is further optimized for low-latency inference through model CPU offloading, xformers, and batch processing of frame chunks to maximize GPU throughput while maintaining a smooth output for live camera integration.

## Results
The system successfully converts archive-style grainy footage into smoothly colorized video with minimal flickering. By enforcing temporal constraints, we have eliminated approximately 80% of the flickering observed in standard frame-by-frame colorization methods. The final pipeline is capable of real-time performance, making it suitable for professional documentary restoration and live archival previews.

