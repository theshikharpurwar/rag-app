import sys
import json
import torch
from transformers import Qwen2_5_VLFoConditionalGeneration, AutoProcessor
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

def retrieve_context(question, pdf_id):
    # Simple cosine similarity for retrieval (replace with ANN if needed)
    query_embedding = embed_model(question)  # Assume embed_model is defined or use a text embedder
    vectors = collection.find({"id": {"$regex": pdf_id}})
    
    best_match = None
    max_similarity = -1
    
    for vector in vectors:
        similarity = cosine_similarity(query_embedding, vector['embedding'])
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = vector
    
    return best_match['metadata']['imagePath'] if best_match else None

def cosine_similarity(vec1, vec2):
    # Simple cosine similarity implementation
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

# Load Qwen 2.5-VL model
model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLFoConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    cache_dir="./qwen_cache"
)
processor = AutoProcessor.from_pretrained(model_path, cache_dir="./qwen_cache")

if __name__ == "__main__":
    question = sys.argv[1]
    pdf_id = sys.argv[2]
    
    # Retrieve relevant context (image path)
    image_path = retrieve_context(question, pdf_id)
    
    if not image_path:
        print("No relevant context found")
        sys.exit(1)
    
    # Prepare messages for Qwen 2.5-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Answer the user query: {question}, based on the image provided."}
            ]
        }
    ]
    
    # Process and generate response
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    print(output_text[0])