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

def query_and_generate(query, image_path):
    query_embedding = colpali_embed_model(query)
    result = client.search(collection_name, query_embedding, {"limit": 1})

    if not result["hits"]:
        return "No relevant images found."

    image_id = result["hits"][0]["id"]
    image_payload = result["hits"][0]["payload"]["image"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Answer the user query: {query}, based on the image provided."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False, streaming=True)

    return output_text[0]