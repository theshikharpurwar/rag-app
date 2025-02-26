# python/llm/qwen_llm.py
import logging
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

class QwenVLModel:
    """Qwen 2.5 Vision-Language model implementation"""

    def __init__(self, model_path, params=None):
        self.model_path = model_path
        self.params = params or {}
        self._load_model()

    def _load_model(self):
        """Load the model and processor"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Qwen model from {self.model_path} on {device}")

            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                attn_implementation="flash_attention_2" if device == "cuda" else None,
                device_map="auto" if device == "cuda" else None
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            logger.info("Qwen model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Qwen model: {e}")
            raise

    def generate_response(self, query, image_paths):
        """Generate a response based on the query and images"""
        try:
            if not image_paths:
                return "No relevant images found for this query."

            # Build messages with images
            messages = [{"role": "user", "content": []}]

            # Add images to message
            for img_path in image_paths:
                messages[0]["content"].append({
                    "type": "image",
                    "image": img_path
                })

            # Add text query
            messages[0]["content"].append({
                "type": "text",
                "text": f"Question: {query}\nPlease analyze these document images and provide a detailed answer."
            })

            # Process with Qwen
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)

            # Extract generation parameters
            max_new_tokens = self.params.get("max_new_tokens", 512)
            temperature = self.params.get("temperature", 0.7)
            do_sample = self.params.get("do_sample", True)

            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature
                )

            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            return output_text[0]

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise