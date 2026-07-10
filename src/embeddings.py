"""
embeddings.py

Purpose:
Convert text chunks into numerical vectors (embeddings)
using HuggingFace Sentence Transformer model.

These embeddings are used by FAISS for similarity search.
"""


from langchain_community.embeddings import HuggingFaceEmbeddings



# --------------------------------------------------
# Create Embedding Model
# --------------------------------------------------

def get_embeddings():

    """
    Load embedding model.

    Model:
    sentence-transformers/all-MiniLM-L6-v2

    Dimension:
    384

    Returns:
    HuggingFace Embedding object
    """


    embeddings = HuggingFaceEmbeddings(

        model_name=
        "sentence-transformers/all-MiniLM-L6-v2",


        model_kwargs={
            "device": "cpu"
        },


        encode_kwargs={
            "normalize_embeddings": True
        }

    )


    return embeddings