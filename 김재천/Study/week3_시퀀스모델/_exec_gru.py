import nbformat
from nbclient import NotebookClient

path = '/sessions/magical-lucid-einstein/mnt/finance_project/김재천/Study/week3_시퀀스모델/03_GRU_실습.ipynb'
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=300, kernel_name='python3',
                        resources={'metadata': {'path': '/sessions/magical-lucid-einstein/mnt/finance_project/김재천/Study/week3_시퀀스모델'}})
client.execute()
nbformat.write(nb, path)

print('cells:', len(nb.cells))
err = 0
stderr_cnt = 0
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        for o in c.get('outputs', []):
            if o.get('output_type') == 'error':
                err += 1
                print('[ERR cell', i, ']', o.get('ename'), ':', o.get('evalue'))
            if o.get('output_type') == 'stream' and o.get('name') == 'stderr':
                stderr_cnt += 1
                print('[STDERR cell', i, ']', o.get('text', '')[:400])
print('errors:', err)
print('stderr_streams:', stderr_cnt)
