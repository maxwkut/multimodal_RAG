import json
import os
import sys
from typing import List

import lancedb
import torch
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from multimodal_lancedb import MultimodalLanceDB
from utils import display_retrieved_results, load_json_file


def get_text_embedding(text: str):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**text_inputs)

    if isinstance(text_embedding, torch.Tensor):
        embedding = text_embedding.detach().cpu().numpy()
    return embedding[0]


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


if __name__ == "__main__":
    # lancedb setup
    LANCEDB_HOST_FILE = ".lancedb"
    TBL_NAME = "test_tbl"
    db = lancedb.connect(LANCEDB_HOST_FILE)

    # load metadata files
    vid1_metadata_path = "data/videos/metadata.json"
    vid1_metadata = load_json_file(vid1_metadata_path)

    # collect transcripts and image paths
    vid1_trans = [vid["transcript"] for vid in vid1_metadata]
    vid1_img_path = [vid["extracted_frame_path"] for vid in vid1_metadata]

    # Update transcripts to include n neighboring chunks
    # for video1, we pick n = 7
    n = 7
    updated_vid1_trans = [
        " ".join(vid1_trans[i - int(n / 2) : i + int(n / 2)])
        if i - int(n / 2) >= 0
        else " ".join(vid1_trans[0 : i + int(n / 2)])
        for i in range(len(vid1_trans))
    ]

    # also need to update the updated transcripts in metadata
    for i in range(len(updated_vid1_trans)):
        vid1_metadata[i]["transcript"] = updated_vid1_trans[i]

    # Ingest into LanceDB
    embedder = CLIPEmbeddings()

    _ = MultimodalLanceDB.from_text_image_pairs(
        texts=updated_vid1_trans,
        image_paths=vid1_img_path,
        embedding=embedder,
        metadatas=vid1_metadata,
        connection=db,
        table_name=TBL_NAME,
        mode="overwrite",
    )

    # Creating a LanceDB vector store
    vectorstore = MultimodalLanceDB(
        uri=LANCEDB_HOST_FILE, embedding=embedder, table_name=TBL_NAME
    )

    # ask to return top 3 most similar documents
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    query1 = "a ship"
    results = retriever.invoke(query1)
    # display_retrieved_results(results)
