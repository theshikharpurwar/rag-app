# python/compute_embeddings.py
import sys
import json
from sentence_transformers import SentenceTransformer

def main():
    input_text = sys.stdin.read().strip()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([input_text])
    print(json.dumps(embeddings.tolist()))

if __name__ == '__main__':
    main()