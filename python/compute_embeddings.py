import sys
import os
import json
from pdf2image import convert_from_path
from colpali_engine.models import ColPali, ColPaliProcessor
import pymongo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv('MONGODB_URI')
DB_NAME = 'rag_db'
COLLECTION_NAME = 'vector_store'

# Connect to MongoDB
client = pymongo.MongoClient(MONGODB_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

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
        embedding = embed_model(processed_image).tolist()  # Convert to list for JSON serialization
        embeddings.append({
            "id": f"page_{page_number}_{os.path.basename(pdf_path)}",
            "embedding": embedding,
            "metadata": {
                "imagePath": f"page_{page_number}.jpg",  # Save or reference image path
                "pageNumber": page_number
            }
        })

    # Store in MongoDB
    collection.insert_many(embeddings)
    return json.dumps(embeddings)

if __name__ == "__main__":
    pdf_path = sys.argv[1]
    result = process_pdf(pdf_path)
    print(result)