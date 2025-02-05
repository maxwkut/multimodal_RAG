import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def get_joint_embedding(image_path: str, text: str):
    # Preprocessing
    image = Image.open(image_path)
    image_inputs = clip_processor(images=image, return_tensors="pt")
    text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True)

    # Generate embeddings
    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**image_inputs)
        text_embedding = clip_model.get_text_features(**text_inputs)
    joint_embedding = torch.cat([image_embedding, text_embedding], dim=-1)

    return joint_embedding


if __name__ == "__main__":
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # get_joint_embedding(args)
