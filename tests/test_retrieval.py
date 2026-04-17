import ast, json, os, shutil

ast.parse(open('app/services/retrieval.py', encoding='utf-8').read())
print('Syntax OK')

from app.services.retrieval import RetrievalService

svc = RetrievalService()

# Setup mock storage
os.makedirs('data/embeddings/chunks', exist_ok=True)

with open('data/embeddings/files.json', 'w') as f:
    json.dump({
        'finance.txt':  {'embedding': [1.0, 0.0, 0.0]},
        'python.txt':   {'embedding': [0.0, 1.0, 0.0]},
        'history.txt':  {'embedding': [0.0, 0.0, 1.0]},
    }, f)

for name, embs in [
    ('finance.txt',  [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0]]),
    ('python.txt',   [[0.1, 0.85, 0.05], [0.0, 0.9, 0.1]]),
    ('history.txt',  [[0.05, 0.0, 0.95], [0.0, 0.1, 0.9]]),
]:
    with open('data/embeddings/chunks/' + name + '.json', 'w') as f:
        json.dump([{'text': 'chunk from ' + name, 'embedding': e} for e in embs], f)

# Query pointing toward finance
query_emb = [0.95, 0.05, 0.0]

# Stage 1: should pick finance.txt first, python.txt second
top_files = svc._get_top_files(query_emb, top_k=2)
print('Stage 1 - top files:', top_files)

# Stage 2: top chunks from selected files
top_chunks = svc._get_top_chunks(query_emb, top_files, top_k=3)
print('Stage 2 - top chunks:')
for c in top_chunks:
    print('  score=' + str(round(c['score'], 4)) + ' | file=' + c['file_name'] + ' | text=' + c['text'])

# Cleanup
shutil.rmtree('data/embeddings')
print('Cleanup done.')
