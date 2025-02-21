import torch
from transformers import ColPaliProcessor, ColPaliModel

def compute_embeddings(file):
    model_name = "vidore/colpali-v1.2"
    embed_model = ColPaliModel.from_pretrained(model_name)
    processor = ColPaliProcessor.from_pretrained(model_name)

    embeddings = []
    for image in file:  # Assuming file is a list of image paths
        image = processor.process_images(image)
        embedding = embed_model(image)[0].detach().numpy().tolist()
        embeddings.append(embedding)

    return embeddings