import sys
import os
import lancedb
from embeddings import CLIPEmbeddings
from multimodal_lancedb import MultimodalLanceDB


def retrieve_results(query: str, table_name: str):
    """Retrieve results based on query from vectorstore.

    Args:
        query (str)
        table_name (str)

    Returns:
        _type_: _description_
    """
    db = lancedb.connect(".lancedb")

    embedder = CLIPEmbeddings()
    # Creating a LanceDB vector store
    vectorstore = MultimodalLanceDB(
        uri=".lancedb", embedding=embedder, table_name=table_name
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    results = retriever.invoke(query)
    return results


if __name__ == "__main__":
    TBL_NAME = "test_tbl"

    results = retrieve_results(query="what does a cow say to a cat?", table_name=TBL_NAME)
    print(results)