# python/compute_embeddings.py
import sys
import json
from byaldi import RAGMultiModalModel

def main():
    raw_input = sys.stdin.read()
    input_text = str(raw_input).strip()
    cleaned_text = " ".join(input_text.split())
    
    if len(cleaned_text) < 10:
        print(json.dumps([]))
        sys.stdout.flush()
        return

    try:
        RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
        embedding = RAG.encode(cleaned_text)
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        print(json.dumps(embedding))
        sys.stdout.flush()
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.stdout.flush()

if __name__ == '__main__':
    main()