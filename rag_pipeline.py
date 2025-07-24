# === rag_pipeline.py ===

import os
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document


# === Load and Combine Knowledge CSVs ===
def load_knowledgebase():
    files = [
        "knowledge_training_data.csv",
        "shortfall_test_predictions.csv",
        "knowledge_batch_predictions.csv"
    ]
    docs = []
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                content = " | ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
                docs.append(Document(page_content=content))
    return docs

# === Embed and Save FAISS Index ===
def build_faiss_index():
    docs = load_knowledgebase()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_store")

    print("✅ FAISS index built and saved.")

if __name__ == "__main__":
    build_faiss_index()
