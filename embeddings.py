from typing import List, Optional

import torch
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


class CLIPEmbeddings(BaseModel, Embeddings):
    """CLIP embedding model using Hugging Face transformers"""
    
    # Define these as class attributes for Pydantic
    clip_model: Optional[CLIPModel] = None
    clip_processor: Optional[CLIPProcessor] = None
    model_name: str = "openai/clip-vit-base-patch32"
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        """Initialize the CLIP model and processor once."""
        super().__init__(**kwargs)
        # Load the model and processor during initialization
        if self.clip_model is None:
            self.clip_model = CLIPModel.from_pretrained(self.model_name)
        if self.clip_processor is None:
            self.clip_processor = CLIPProcessor.from_pretrained(self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using CLIP.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        # Process in a single batch for efficiency
        text_inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embeddings = self.clip_model.get_text_features(**text_inputs)
        
        # Convert tensor to list of numpy arrays
        embeddings = text_embeddings.detach().cpu().numpy().tolist()
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
        self, texts: List[str], images: List[str], batch_size=8
    ) -> List[List[float]]:
        """Embed a list of image-text pairs using CLIP.

        Args:
            texts: The list of texts to embed.
            images: The list of paths to images to embed.
            batch_size: Batch size for processing (default: 8).

        Returns:
            List of embeddings, one for each image-text pair.
        """
        assert len(texts) == len(images), "Number of texts must match number of images."
        
        all_embeddings = []
        total_pairs = len(texts)
        
        # Process in batches
        for i in tqdm(range(0, total_pairs, batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_image_paths = images[i:i+batch_size]
            
            # Load and preprocess all images in the batch
            batch_images = [Image.open(img_path) for img_path in batch_image_paths]
            
            # Process images and texts
            image_inputs = self.clip_processor(images=batch_images, return_tensors="pt", padding=True)
            text_inputs = self.clip_processor(text=batch_texts, return_tensors="pt", padding=True)
            
            # Generate embeddings
            with torch.no_grad():
                image_embeddings = self.clip_model.get_image_features(**image_inputs)
                text_embeddings = self.clip_model.get_text_features(**text_inputs)
                
                # Calculate joint embeddings
                joint_embeddings = (image_embeddings + text_embeddings) / 2
                
            # Convert to numpy and add to results
            batch_embeddings = joint_embeddings.detach().cpu().numpy()
            all_embeddings.extend(batch_embeddings.tolist())
            
        return all_embeddings


# These standalone functions are now deprecated since the functionality is in the class
def get_text_embedding(text: str):
    """Deprecated: Use CLIPEmbeddings.embed_query instead"""
    embedder = CLIPEmbeddings()
    return embedder.embed_query(text)


def get_joint_embedding(image_path: str, text: str):
    """Deprecated: Use CLIPEmbeddings.embed_image_text_pairs instead"""
    embedder = CLIPEmbeddings()
    return embedder.embed_image_text_pairs([text], [image_path])[0]