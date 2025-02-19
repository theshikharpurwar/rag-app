# python/compute_embeddings.py
import sys
import json
from sentence_transformers import SentenceTransformer

def main():
    # Read the text from standard input.
    input_text = sys.stdin.read().strip()
    if not input_text:
        # If there is no text, output an empty list.
        print(json.dumps([]))
        return

    # Load a pre-trained embedding model.
    # 'all-MiniLM-L6-v2' is a lightweight model ideal for many embedding tasks.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Compute the embedding for the input text.
    embedding = model.encode(input_text).tolist()
    
    # Output the embedding as a JSON array.
    print(json.dumps(embedding))

if __name__ == '__main__':
    main()
