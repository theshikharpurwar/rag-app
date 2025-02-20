# python/compute_embeddings.py
import sys
import json
import warnings
from sentence_transformers import SentenceTransformer

def main():
    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # Read input from stdin.
    raw_input = sys.stdin.read()
    input_text = str(raw_input).strip()
    cleaned_text = " ".join(input_text.split())

    if len(cleaned_text) < 10:
        print(json.dumps([]))
        sys.stdout.flush()
        return

    try:
        # Load the model (this may take a moment on first run).
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Pass the cleaned text directly.
        # Increase max_length to a higher value to avoid the warning
        embedding = model.encode([cleaned_text], max_length=512)
        # If the embedding is a numpy array, convert it to a list.
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        print(json.dumps(embedding))
        sys.stdout.flush()
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.stdout.flush()

if __name__ == '__main__':
    main()