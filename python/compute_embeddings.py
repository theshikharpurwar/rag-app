import sys
import os
import json
from pdf2image import convert_from_path
from colpali_engine.models import ColPali, ColPaliProcessor

def process_pdf(pdf_path):
    # Convert PDF to images (one per page)
    images = convert_from_path(pdf_path)
    embeddings = []

    # Load ColPali model and processor
    model_name = "vidore/colpali-v1.2"
    embed_model = ColPali.from_pretrained(model_name)
    processor = ColPaliProcessor.from_pretrained(model_name)

    # Process each page
    for page_number, image in enumerate(images, 1):
        # Preprocess image
        processed_image = processor.process_images(image)
        # Generate embedding
        embedding = embed_model(processed_image).tolist()  # Convert to list for JSON
        embeddings.append({
            "id": f"page_{page_number}_{os.path.basename(pdf_path)}",
            "embedding": embedding,
            "metadata": {
                "imagePath": f"page_{page_number}.jpg",  # Save or reference image path
                "pageNumber": page_number
            }
        })

    # Read existing vectors, append new ones, and write back
    vector_store_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend', 'vector_store.json')
    if os.path.exists(vector_store_path):
        with open(vector_store_path, 'r') as f:
            existing_vectors = json.load(f)
    else:
        existing_vectors = []
    
    updated_vectors = existing_vectors + embeddings
    with open(vector_store_path, 'w') as f:
        json.dump(updated_vectors, f, indent=2)

    return json.dumps(embeddings)

if __name__ == "__main__":
    pdf_path = sys.argv[1]
    result = process_pdf(pdf_path)
    print(result)