This project is part of the Gen AI Internship at Elevance Skills.

# Task 2: Colorization of Historical Photographs

## Problem Statement
Historical photographs, often recorded in black-and-white or sepia, carry invaluable cultural and personal history. Traditional manual colorization is a labor-intensive process requiring deep historical knowledge. This task aims to automate the restoration of historical photos using Generative AI, focusing on era-specific accuracy such as WWII, the Victorian era, or the roaring 1920s.

## Dataset
The project utilized a curated collection of historical archival photos ranging from 1920s urban scenes to 1940s WWII military photography. High-resolution reference samples of authentic Technicolor and Kodachrome film were used to calibrate the model for period-accurate color reproduction.

## Methodology
To achieve historical accuracy, we leverage Stable Diffusion v1.5 coupled with ControlNet (Canny/Lineart). Methodology includes era-specific prompting using precise historical descriptors to guide the generative process. ControlNet ensures that uniform details, medals, and architectural features are perfectly preserved while the AI avoids modern, over-saturated colors in favor of authentic, muted tones characteristic of vintage film technology.

## Results
The system demonstrates high performance in restoring archival footage with a result that feels authentic to the period. Historical accuracy was validated by comparing the output palettes with actual film samples from the target eras, showing a significant improvement over generic colorization methods in preserving the 'antique' feel of the original source.

