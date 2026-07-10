# 📚 ResearchGPT: Hybrid RAG Research Paper Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange)
![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-purple)


## 📌 Overview

ResearchGPT is an AI-powered **Multi-Document Research Paper Assistant** built using an advanced **Hybrid Retrieval-Augmented Generation (Hybrid RAG)** architecture.

The application allows users to upload research papers or PDF documents and interact with them through a conversational chatbot.

It combines:

- Dense semantic retrieval using FAISS
- Sparse keyword retrieval using BM25
- Cross Encoder Re-ranking
- Google Gemini LLM generation

to provide accurate, context-aware answers with source document references.


---

# 🚀 Features

✔ Upload multiple PDF documents  
✔ Automatic document processing  
✔ Real-time processing progress indicator  
✔ Add new documents without restarting application  
✔ Continue conversation with previous documents  
✔ Hybrid Retrieval System  
✔ Source citations with page numbers  
✔ Conversational chatbot interface  
✔ Detailed AI-generated explanations  


---

# 🧠 Hybrid RAG Architecture


```
                 User Query
                      |
                      ↓
              Hybrid Retriever

          /                    \

         ↓                      ↓

 FAISS Dense Search       BM25 Sparse Search

 (Semantic Search)        (Keyword Search)


          \                    /

                   ↓

           Result Combination

                   ↓

        Cross Encoder Re-Ranking

                   ↓

        Top Relevant Documents

                   ↓

             Gemini LLM

                   ↓

      Final Answer + Source Citation

```


---

# 🔍 How It Works


## 1. Document Upload

Users upload PDF research papers through the Streamlit interface.


```
PDF Document
      |
      ↓
Document Loader
```


---

## 2. Text Extraction

PDF content is extracted using LangChain document loaders.


```
Research Paper

       ↓

Extracted Text

```


---

## 3. Text Chunking

Large documents are divided into smaller chunks.


Technology Used:

- RecursiveCharacterTextSplitter


Example:

```

Complete Document

        ↓

Chunk 1

Chunk 2

Chunk 3

...

```


---

## 4. Embedding Generation


Each chunk is converted into numerical vectors.


Embedding Model:

```
sentence-transformers/all-MiniLM-L6-v2
```


Example:

```
Text Chunk

    ↓

[0.23,0.65,0.87...]

```


---

## 5. Vector Storage


Generated embeddings are stored in FAISS.


```

Embeddings

     ↓

FAISS Vector Database

```


FAISS enables fast semantic similarity search.


---

## 6. Hybrid Retrieval


This project does not rely only on vector search.


It combines:


### Dense Retrieval

Using FAISS:

- Understands semantic meaning
- Finds conceptually similar content


Example:

Question:

```
Explain transformers
```

Finds:

```
attention mechanism
encoder decoder architecture
```


---


### Sparse Retrieval


Using BM25:


- Keyword-based retrieval
- Finds exact terms


Example:


Question:

```
Scaled dot product attention
```

BM25 finds exact matching sections.


---


## 7. Re-Ranking


A Cross Encoder model re-ranks retrieved documents.


Model:


```
cross-encoder/ms-marco-MiniLM-L-6-v2
```


Purpose:

- Improves relevance
- Removes weak results
- Sends best context to LLM


---

## 8. Answer Generation


Google Gemini receives:

```
User Question

+

Retrieved Context

```

and generates:

```
Detailed Answer

+

Source Reference

```

---

# 🛠️ Tech Stack


| Component | Technology |
|---|---|
| Programming | Python |
| Application | Streamlit |
| RAG Framework | LangChain |
| LLM | Google Gemini |
| Dense Retrieval | FAISS |
| Sparse Retrieval | BM25 |
| Embeddings | HuggingFace Sentence Transformers |
| Re-Ranking | Cross Encoder |
| Document Processing | PyPDFLoader |


---


# 📂 Project Structure


```

ResearchGPT-Hybrid-RAG

│

├── app.py
│
├── ingest.py
│
├── requirements.txt
│
├── README.md
│
├── .env.example
│
├── .gitignore

│
├── src/

│   ├── loader.py

│   ├── chunking.py

│   ├── embeddings.py

│   ├── vector_store.py

│   ├── hybrid_retriever.py

│   └── rag_chain.py


│
├── data/

│   ├── raw/

│   │    └── .gitkeep

│   │

│   └── processed/

│        └── .gitkeep


```


---

# ⚙️ Installation Guide


## 1. Clone Repository


```bash
git clone https://github.com/jidnyasadthakre07/ResearchGPT-Hybrid-RAG-System
```
Move inside folder:

```bash
cd ResearchGPT-Hybrid-RAG
```


---

## 2. Create Virtual Environment


```bash
python -m venv venv
```


Activate environment:


Windows:


```bash
venv\Scripts\activate
```


Linux/Mac:


```bash
source venv/bin/activate
```


---

## 3. Install Dependencies


```bash
pip install -r requirements.txt
```


---

# 🔑 Environment Setup


Create a `.env` file:


```
GOOGLE_API_KEY=your_google_gemini_api_key
```


---

# ▶️ Run Application


Start Streamlit server:


```bash
streamlit run app.py
```


Open browser:


```
http://localhost:8501
```


---

# 💬 Example Questions


After uploading documents:


```
Explain self-attention mechanism
```


```
What is transformer architecture?
```


```
Compare BERT and GPT models
```


```
Summarize this research paper
```


---

# 📸 Application Screenshots


## Home Page

(Add screenshot here)


## Document Processing

(Add screenshot here)


## Chat Interface

(Add screenshot here)


---


# 📊 Results


The system provides:


✔ Detailed explanations

✔ Context-based responses

✔ Document references

✔ Page number citations


Example:


Question:

```
What is self-attention?
```


Answer:


```
Self-attention is a mechanism used in transformer architecture
that allows tokens to understand relationships with other tokens.
```


Source:


```
Paper: attention.pdf

Page: 2
```


---

# 🚀 Future Enhancements


- RAGAS Evaluation Metrics
- Query Rewriting
- Agentic RAG
- Knowledge Graph RAG
- Multi-format document support
- User authentication
- Cloud Deployment


---

# 👩‍💻 Author


**Jidnyasa Thakre**

AI & Machine Learning Enthusiast

Focus Areas:

- Machine Learning
- Deep Learning
- Generative AI
- Large Language Models
- Retrieval-Augmented Generation


---
