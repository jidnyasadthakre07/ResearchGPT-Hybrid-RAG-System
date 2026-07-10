"""
vector_store.py

Purpose:
Create and load FAISS vector database
"""


from langchain_community.vectorstores import FAISS

from src.embeddings import get_embeddings



# Create FAISS database

def create_vectorstore(chunks):


    embeddings = get_embeddings()


    db = FAISS.from_documents(

        documents=chunks,

        embedding=embeddings

    )


    db.save_local(

        "vectorstore"

    )


    print(
        "FAISS index saved successfully"
    )


    return db




# Load FAISS database

def load_vectorstore():


    embeddings = get_embeddings()


    db = FAISS.load_local(

        "vectorstore",

        embeddings,

        allow_dangerous_deserialization=True

    )


    return db