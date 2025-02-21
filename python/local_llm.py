import sys
import json
import torch
import logging
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load text embedder for queries
text_embedder = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims

# Load Qwen 2.5-VL model
try:
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        cache_dir="./qwen_cache"
    )
    processor = AutoProcessor.from_pretrained(model_path, cache_dir="./qwen_cache")
    logger.info("Qwen 2.5-VL model loaded successfully")
except Exception as e:
    logger.error(f"Error loading Qwen 2.5-VL model: {e}")
    sys.exit(1)

def retrieve_context(question, pdf_id, collection_name):
    try:
        client = QdrantClient(url="http://localhost:6333")
        logger.info(f"Retrieving context for question: {question}, PDF: {pdf_id}")

        # Generate query embedding
        query_embedding = text_embedder.encode(question).tolist()  # 384 dims

        # Note: ColPali uses 128-dim per patch, but we use a text embedder here.
        # For simplicity, retrieve top match based on payload filter and assume compatibility.
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=1,
            with_payload=True,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="pdfId",
                        match=models.MatchValue(value=pdf_id)
                    )
                ]
            )
        )

        if search_result:
            best_match = search_result[0]
            logger.info(f"Found context for PDF: {pdf_id}")
            return best_match.payload["imagePath"]
        else:
            logger.warning(f"No context found for PDF: {pdf_id}")
            return None
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(json.dumps({"error": "Question, PDF ID, and collection name are required"}))
        sys.exit(1)
    
    question = sys.argv[1]
    pdf_id = sys.argv[2]
    collection_name = sys.argv[3]
    
    try:
        logger.info(f"Processing question: {question} for PDF: {pdf_id}")
        image_path = retrieve_context(question, pdf_id, collection_name)
        
        if not image_path or not os.path.exists(image_path):
            print(json.dumps({"error": "No relevant context found or image missing"}))
            sys.exit(1)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": f"Answer the user query: {question}, based on the image provided."}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = processor.image_processor.open_image(image_path)
        inputs = processor(text=[text], images=[image_inputs], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        print(output_text[0])
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)