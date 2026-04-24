"""실행된 노트북의 셀별 출력을 검토용으로 덤프."""
import json
import nbformat

nb = nbformat.read('04_PyTorch_학습루프_실습.ipynb', as_version=4)

def short(text, max_len=800):
    if len(text) > max_len:
        return text[:max_len] + f'\n... [truncated, total {len(text)} chars]'
    return text

code_idx = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != 'code':
        continue
    code_idx += 1
    src_preview = c.source.split('\n', 1)[0][:80]
    print(f'\n{"="*78}')
    print(f'[code cell #{code_idx}] source first line: {src_preview}')
    if not c.outputs:
        print('  (no output)')
        continue
    for out in c.outputs:
        otype = out.get('output_type', '')
        if otype == 'stream':
            print(f'  STREAM({out.get("name")}):')
            print(short(out.get('text', '')))
        elif otype == 'execute_result':
            data = out.get('data', {})
            if 'text/plain' in data:
                print(f'  RESULT(text/plain):')
                print(short(data['text/plain']))
            if 'text/html' in data:
                print(f'  RESULT(html, {len(data["text/html"])} chars) [DataFrame 렌더 포함]')
        elif otype == 'display_data':
            data = out.get('data', {})
            kinds = [k for k in data if k not in ('text/plain',)]
            print(f'  DISPLAY_DATA (mime: {kinds})')
            if 'text/plain' in data:
                print(short(data['text/plain'], 200))
        elif otype == 'error':
            print(f'  ERROR: {out.get("ename")}: {out.get("evalue")}')
            print('  traceback:')
            for line in out.get('traceback', []):
                print('   ', line[:200])
