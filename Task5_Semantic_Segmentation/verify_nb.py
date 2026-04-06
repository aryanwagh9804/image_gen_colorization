import json

with open('semantic_segmentation.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print(f'Total cells: {len(nb["cells"])}')
for i, c in enumerate(nb['cells']):
    src_preview = ''.join(c.get('source', []))[:60].replace('\n', ' ')
    print(f'  [{i}] {c["cell_type"]:8s}  id={c["id"]:30s}  {src_preview}...')
