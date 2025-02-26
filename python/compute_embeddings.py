import sys
import os
import json
import logging
from pdf2image import convert_from_path, pdfinfo_from_path
from colpali_engine.models import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient, models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_pdf(pdf_path, collection_name):
    try:
        logger.info(f"Processing PDF: {pdf_path}")
        poppler_path = r"C:\poppler-24.08.0\Library\bin"
        logger.info("Getting PDF page count...")
        pdf_info = pdfinfo_from_path(pdf_path, poppler_path=poppler_path)
        num_pages = pdf_info["Pages"]

        client = QdrantClient(url="http://localhost:6333")
        model_name = "vidore/colpali-v1.2"
        cache_dir = "D:\\rag-app\\cache"
        logger.info(f"Loading ColPali model from {cache_dir}...")
        try:
            embed_model = ColPali.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True,
                force_download=False,
                torch_dtype="float16"  # Reduce memory usage
            )
            processor = ColPaliProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True,
                force_download=False
            )
        except Exception as e:
            logger.error(f"Failed to load ColPali model: {e}")
            raise
        logger.info("ColPali model loaded successfully")

        points = []
        for page_number in range(1, num_pages + 1):
            try:
                logger.info(f"Processing page {page_number}/{num_pages}")
                images = convert_from_path(
                    pdf_path,
                    poppler_path=poppler_path,
                    first_page=page_number,
                    last_page=page_number
                )
                image = images[0]
                image_path = f"uploads/page_{page_number}_{os.path.basename(pdf_path)}.jpg"
                image.save(image_path, "JPEG")

                processed_image = processor.process_images([image])[0]
                embeddings = embed_model(processed_image.unsqueeze(0)).squeeze(0).tolist()

                point = models.PointStruct(
                    id=f"page_{page_number}_{os.path.basename(pdf_path)}",
                    vector=embeddings,
                    payload={
                        "imagePath": image_path,
                        "pageNumber": page_number,
                        "pdfId": os.path.basename(pdf_path)
                    }
                )
                points.append(point)
                del image, processed_image, images  # Free memory
            except Exception as e:
                logger.error(f"Error processing page {page_number}: {e}")
                continue

        logger.info("Upserting points to Qdrant...")
        client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Embeddings stored in Qdrant for PDF: {pdf_path}")
        return json.dumps({"message": "Embeddings stored successfully"})
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

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