"""
loader.py

Purpose:
Load all research paper PDFs
from data/raw folder.
"""


import os

from langchain_community.document_loaders import PyPDFLoader



def load_pdfs(folder_path):


    documents = []


    for filename in os.listdir(folder_path):


        if filename.endswith(".pdf"):


            pdf_path = os.path.join(
                folder_path,
                filename
            )


            print(
                f"Loading: {filename}"
            )


            loader = PyPDFLoader(
                pdf_path
            )


            pages = loader.load()


            documents.extend(
                pages
            )


    return documents