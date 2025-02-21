from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from colpali_engine import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient, models

# Load Qwen2.5-VL model and processor
model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_path)

# Load ColPali model and processor
colpali_model_name = "vidore/colpali-v1.2"
colpali_embed_model = ColPali.from_pretrained(colpali_model_name)
colpali_processor = ColPaliProcessor.from_pretrained(colpali_model_name)

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333")
collection_name = "deepseek-colpali-multimodalRAG"

# Create collection if it doesn't exist
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        on_disk_payload=True,
        vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE, on_disk=True),
        multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
    )

def embed_image(image_path):
    image = colpali_processor.process_images(image_path)
    return colpali_embed_model(image)

def embed_text(text):
    return colpali_embed_model(text)

def store_embeddings(embeddings):
    points = [{"id": i, "vector": embedding, "payload": {"image": f"image_{i}"}} for i, embedding in enumerate(embeddings)]
    client.upsert(collection_name=collection_name, points=points)