"""
hybrid_retriever.py

Purpose:
Hybrid Retrieval

FAISS (Dense Search)
+
BM25 (Sparse Search)
+
Cross Encoder Re-ranking
"""


from langchain_classic.retrievers import EnsembleRetriever

from langchain_community.retrievers import BM25Retriever


from sentence_transformers import CrossEncoder



# =================================
# Create Hybrid Retriever
# =================================


def create_hybrid_retriever(
        vectorstore,
        chunks
):


    # Dense Retriever
    # FAISS Vector Search

    faiss_retriever = vectorstore.as_retriever(

        search_kwargs={
            "k": 10
        }

    )



    # Sparse Retriever
    # BM25 Keyword Search

    bm25_retriever = BM25Retriever.from_documents(

        chunks

    )


    bm25_retriever.k = 10



    # Combine FAISS + BM25

    hybrid_retriever = EnsembleRetriever(

        retrievers=[

            faiss_retriever,

            bm25_retriever

        ],


        weights=[

            0.5,

            0.5

        ]

    )



    return hybrid_retriever




# =================================
# Cross Encoder Re-ranking
# =================================


def rerank_documents(
        query,
        documents
):


    reranker = CrossEncoder(

        "cross-encoder/ms-marco-MiniLM-L-6-v2"

    )



    pairs = []


    for doc in documents:


        pairs.append(

            [

                query,

                doc.page_content

            ]

        )



    scores = reranker.predict(

        pairs

    )



    ranked_docs = sorted(

        zip(
            scores,
            documents
        ),

        reverse=True,

        key=lambda x: x[0]

    )



    final_docs = [

        doc

        for score, doc in ranked_docs[:5]

    ]



    return final_docs