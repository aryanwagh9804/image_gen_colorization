import json
import os

def patch_notebook(path):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'demo.launch' in "".join(cell['source']):
            # Add .queue() and improve launch message
            new_source = []
            for line in cell['source']:
                if 'demo.launch' in line:
                    new_source.append("demo.queue().launch(share=True)\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            break
            
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

patch_notebook(r'd:\Downloads\image_gen_colorization\Task3_Conditional_Colorization\conditional_colorization.ipynb')
patch_notebook(r'd:\Downloads\image_gen_colorization\Task2_Historical_Colorization\historical_colorization.ipynb')

print("Gradio Queue enabled in both Task 2 and Task 3!")
