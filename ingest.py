"""
Create FAISS Vector Database

Run this file once before starting Streamlit app
"""


from src.loader import load_pdfs

from src.chunking import create_chunks

from src.vector_store import create_vectorstore



DATA_PATH = "data/raw"



def main():


    print("Loading PDFs...")


    documents = load_pdfs(
        DATA_PATH
    )


    print(
        f"Documents loaded: {len(documents)}"
    )



    print("Creating chunks...")


    chunks = create_chunks(
        documents
    )


    print(
        f"Chunks created: {len(chunks)}"
    )



    print("Creating FAISS Vector Store...")


    create_vectorstore(
        chunks
    )


    print(
        "Vector Database Created Successfully!"
    )




if __name__ == "__main__":

    main()