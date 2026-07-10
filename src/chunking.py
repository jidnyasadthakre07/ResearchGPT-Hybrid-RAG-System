"""
chunking.py

Purpose:
Split large research paper text into smaller chunks
for RAG processing.
"""


from langchain_text_splitters import RecursiveCharacterTextSplitter



def create_chunks(documents):

    """
    Convert PDF documents into smaller chunks
    """

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200,

        length_function=len

    )


    chunks = splitter.split_documents(
        documents
    )


    print(
        f"Total chunks created: {len(chunks)}"
    )


    return chunks