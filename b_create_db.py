import lancedb

from embeddings import CLIPEmbeddings
from multimodal_lancedb import MultimodalLanceDB
from utils import load_json_file


def load_and_transform_chunks(metadata_path: str):
    """
    Loads transcripts and frame from video chunks and transforms transcripts to have some overlap with neighbors.

    Args:
        metadata_path: str
    Returns:
    """

    metadata = load_json_file(metadata_path)
    transcripts = [vid["transcript"] for vid in metadata]
    frame_paths = [vid["extracted_frame_path"] for vid in metadata]

    # Update transcripts to include n neighboring chunks
    n = 7
    updated_transcripts = [
        " ".join(transcripts[i - int(n / 2) : i + int(n / 2)])
        if i - int(n / 2) >= 0
        else " ".join(transcripts[0 : i + int(n / 2)])
        for i in range(len(transcripts))
    ]

    # update transcripts in metadata
    for i, j in enumerate(metadata):
        metadata[i]["transcript"] = updated_transcripts[i]
    
    return metadata, updated_transcripts, frame_paths


def store_embeddings(
    metadata: dict, transcripts: str, frame_paths: str, table_name: str
):
    """Create embeddings and store them in lancedb

    Args:
        metadata (dict): Metadata of video chunks
        transcripts (str): Transcipts for video chunks
        frame_paths (str): Paths to where frames are stored
        table_name (str): Name of table in lancedb
    """
    # Set up db connection
    db = lancedb.connect(".lancedb")

    embedder = CLIPEmbeddings()

    # Store embeddings in db
    _ = MultimodalLanceDB.from_text_image_pairs(
        texts=transcripts,
        image_paths=frame_paths,
        embedding=embedder,
        metadatas=metadata,
        connection=db,
        table_name=table_name,
        mode="overwrite",
    )
    return


if __name__ == "__main__":
    TBL_NAME = "test_tbl"

    metadata, transcripts, frame_paths = load_and_transform_chunks(
        metadata_path="data/videos/metadata.json"
    )
    store_embeddings(
        transcripts=transcripts,
        frame_paths=frame_paths,
        metadata=metadata,
        table_name=TBL_NAME,
    )