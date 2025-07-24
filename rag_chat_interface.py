# import streamlit as st
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chat_models import ChatOllama
# from langchain.chains import RetrievalQA

# # === Streamlit UI ===
# st.set_page_config(page_title="📊 Ask the Supply Chain Model", layout="wide")
# st.title("💬 Step 5: Ask Questions to Your Supply Chain Model (LLaMA 3)")

# st.markdown("""
# Use natural language to ask questions about:
# - Vendor performance
# - Route risk
# - Shortfall causes
# - Congestion impact
# - Model predictions
# """)

# # === Load FAISS Index and LLM ===
# @st.cache_resource
# def load_qa_chain():
#     try:
#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
#         db = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)
#         retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
#         llm = ChatOllama(model="llama3:8b", temperature=0.2)
                
#         qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

#         st.success("✅ FAISS index and LLaMA 3 initialized successfully!")
#         return qa_chain
#     except Exception as e:
#         st.error(f"❌ Failed to initialize RAG system: {e}")
#         return None

# qa_chain = load_qa_chain()

# # === Ask a Question ===
# if qa_chain:
#     user_query = st.text_input("💬 Ask your question:")
#     if user_query:
#         with st.spinner("🔎 Thinking..."):
#             try:
#                 result = qa_chain(user_query)
#                 st.markdown("### ✅ Answer")
#                 st.write(result['result'])

#                 with st.expander("📄 Sources"):
#                     for doc in result['source_documents']:
#                         st.markdown(f"`...{doc.page_content[:400]}...`")
#             except Exception as e:
#                 st.error(f"Error generating answer: {e}")

