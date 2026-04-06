This project is part of the Gen AI Internship at Elevance Skills.

# Task 1: Artistic Style Transfer in Colorization

## Problem Statement
Traditional image colorization aims for realism. However, in creative domains, there is a need to colorize grayscale images while simultaneously applying specific artistic styles such as Post-Impressionism, Sketch, or specific artist styles like Van Gogh. This task involves building a Generative AI pipeline that can transform a black-and-white photo into a stylized, colored artwork.

## Dataset
The project utilized a diverse collection of grayscale urban and nature photography. Benchmarking was performed using a subset of the COCO Val 2017 dataset to ensure that the stylistic transformations did not compromise the structural integrity of common objects.

## Methodology
This task utilizes Stable Diffusion v1.5 integrated with ControlNet (Canny Edge Detection). ControlNet is used to preserve the structural integrity and edges of the original grayscale image, while artistic styles are injected through precise prompt engineering. The workflow involves extracting edge maps, then generating a colorized version that adheres to both the subject structure and the desired artistic style tokens.

## Results
The model successfully colorizes grayscale images while maintaining the original composition with a significant artistic overlay. Results demonstrate high fidelity to edge constraints while delivering vibrant, style-consistent color palettes across watercolor, oil painting, and digital art categories.

