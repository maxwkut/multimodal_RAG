from typing import List

import torch
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


class CLIPEmbeddings(BaseModel, Embeddings):
    """CLIP embedding model using Hugging Face transformers"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using BridgeTower.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        for text in texts:
            embedding = get_text_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text using CLIP.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]

    def embed_image_text_pairs(
        self, texts: List[str], images: List[str], batch_size=2
    ) -> List[List[float]]:
        """Embed a list of image-text pairs using CLIP.

        Args:
            texts: The list of texts to embed.
            images: The list of paths to images to embed.
            batch_size: Batch size for processing (default: 2).

        Returns:
            List of embeddings, one for each image-text pair.
        """
        assert len(texts) == len(images), "Number of texts must match number of images."

        embeddings = []
        for path_to_img, text in tqdm(zip(images, texts), total=len(texts)):
            embedding = get_joint_embedding(path_to_img, text)
            embeddings.append(embedding)
        return embeddings


def get_text_embedding(text: str):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**text_inputs)

    if isinstance(text_embedding, torch.Tensor):
        embedding = text_embedding.detach().cpu().numpy()
        return embedding[0]
    else:
        raise Exception


def get_joint_embedding(image_path: str, text: str):
    """kil;k"""
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Preprocessing
    image = Image.open(image_path)
    image_inputs = clip_processor(images=image, return_tensors="pt")
    text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True)

    # Generate embeddings
    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**image_inputs)
        text_embedding = clip_model.get_text_features(**text_inputs)
    joint_embedding = (image_embedding + text_embedding) / 2

    # Convert tensor to np
    if isinstance(joint_embedding, torch.Tensor):
        embedding = joint_embedding.detach().cpu().numpy()

    return embedding[0]