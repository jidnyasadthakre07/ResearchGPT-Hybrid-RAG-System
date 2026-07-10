"""
rag_chain.py

Purpose:
Hybrid RAG Pipeline

FAISS Dense Retrieval
+
BM25 Sparse Retrieval
+
Cross Encoder Re-ranking
+
Gemini LLM Generation
"""


import os

from dotenv import load_dotenv


from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser


from src.vector_store import load_vectorstore

from src.hybrid_retriever import (
    create_hybrid_retriever,
    rerank_documents
)



load_dotenv()



def get_rag_response(question):


    # ==================================
    # Load FAISS Vector Database
    # ==================================

    db = load_vectorstore()



    # ==================================
    # Get documents from FAISS
    # Needed for BM25
    # ==================================

    chunks = list(
        db.docstore._dict.values()
    )



    # ==================================
    # Create Hybrid Retriever
    # FAISS + BM25
    # ==================================

    hybrid_retriever = create_hybrid_retriever(

        vectorstore=db,

        chunks=chunks

    )



    # Retrieve documents

    retrieved_docs = hybrid_retriever.invoke(

        question

    )



    # ==================================
    # Re-ranking
    # ==================================

    docs = rerank_documents(

        question,

        retrieved_docs

    )



    # ==================================
    # Create Context
    # ==================================

    context = "\n\n".join(

        doc.page_content

        for doc in docs

    )



    # ==================================
    # Gemini LLM
    # ==================================

    llm = ChatGoogleGenerativeAI(

        model="gemini-2.5-flash",

        temperature=0.2,

        google_api_key=os.getenv(
            "GOOGLE_API_KEY"
        )

    )



    # ==================================
    # Prompt Template
    # ==================================

    prompt = ChatPromptTemplate.from_template(
"""

You are an expert AI Research Assistant.

You answer questions from research papers using Hybrid RAG.

Use ONLY the given context.

Generate a detailed explanation.

Follow these rules:

1. Start with a simple definition.

2. Explain the main concept.

3. Explain the working process step-by-step.

4. Mention algorithms, formulas, architectures,
   or techniques if available.

5. Explain advantages.

6. Explain limitations if available.

7. Use headings and bullet points.

8. Do not hallucinate information.

If the answer is not available in the context,
say:
"I could not find this information in the uploaded documents."


====================
Context:
{context}
====================


Question:

{question}


Detailed Answer:

"""
)



    # ==================================
    # Create Chain
    # ==================================

    chain = (

        prompt

        |

        llm

        |

        StrOutputParser()

    )



    answer = chain.invoke(

        {

            "context": context,

            "question": question

        }

    )



    # ==================================
    # Source Citations
    # ==================================

    sources = []


    for doc in docs:


        sources.append(

            {

                "paper": doc.metadata.get(
                    "source"
                ),


                "page": doc.metadata.get(
                    "page"
                )

            }

        )



    return answer, sources