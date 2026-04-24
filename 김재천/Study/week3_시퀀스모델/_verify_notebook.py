"""노트북 syntax 검증 스크립트."""
import ast
import nbformat

nb = nbformat.read('04_PyTorch_학습루프_실습.ipynb', as_version=4)
md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
code = sum(1 for c in nb.cells if c.cell_type == 'code')
print(f'총 {len(nb.cells)} 셀 (markdown={md}, code={code})')

errors = []
for i, c in enumerate(nb.cells):
    if c.cell_type != 'code':
        continue
    try:
        ast.parse(c.source)
    except SyntaxError as e:
        errors.append((i, str(e), c.source[:120]))

if errors:
    for i, e, preview in errors:
        print(f'SYNTAX ERROR cell#{i}: {e}')
        print(f'  preview: {preview!r}')
else:
    print('OK all code cells parse cleanly')
