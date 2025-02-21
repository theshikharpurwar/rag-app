import sys
import os
import json
import logging
from unpdf import PDF  # Import unpdf instead of pdf2image
from colpali_engine.models import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient, models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_pdf(pdf_path, collection_name):
    try:
        logger.info(f"Processing PDF: {pdf_path}")
        # Use unpdf to open the PDF
        pdf = PDF(pdf_path)
        client = QdrantClient(url="http://localhost:6333")

        model_name = "vidore/colpali-v1.2"
        embed_model = ColPali.from_pretrained(model_name)
        processor = ColPaliProcessor.from_pretrained(model_name)

        points = []
        for page_number, page in enumerate(pdf.pages, 1):  # Iterate over pages
            try:
                logger.info(f"Processing page {page_number}")
                # Extract page as an image (PIL format)
                image = page.to_image()

                # Save image for later use in generation
                image_path = f"uploads/page_{page_number}_{os.path.basename(pdf_path)}.jpg"
                image.save(image_path, "JPEG")

                # Process image with ColPali
                processed_image = processor.process_images([image])[0]  # Single image
                embeddings = embed_model(processed_image.unsqueeze(0)).squeeze(0).tolist()  # Multivector embeddings

                point = models.PointStruct(
                    id=f"page_{page_number}_{os.path.basename(pdf_path)}",
                    vector=embeddings,  # Multivector format
                    payload={
                        "imagePath": image_path,
                        "pageNumber": page_number,
                        "pdfId": os.path.basename(pdf_path)
                    }
                )
                points.append(point)
            except Exception as e:
                logger.error(f"Error processing page {page_number}: {e}")
                continue

        client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Embeddings stored in Qdrant for PDF: {pdf_path}")
        return json.dumps({"message": "Embeddings stored successfully"})
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "PDF path and collection name are required"}))
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    collection_name = sys.argv[2]
    try:
        result = process_pdf(pdf_path, collection_name)
        print(result)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)