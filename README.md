# 📦 Supply Chain Risk Assessment with Smart Query Interface

An end-to-end Streamlit-based ML system that predicts quantity shortfalls in supply chains and enables smart querying over predictions using a ChatGPT-style interface powered by LLaMA 3 and LangChain.

---

## 🚀 Features

- Upload and validate Orders, Vendors, Routes datasets
- Merge files based on user-selected common columns
- Train classification models (Random Forest, XGBoost, SVM, Logistic Regression)
- Predict shortfalls for single or bulk uploaded order files
- Ask supply chain questions using a smart LLM-based chat interface
- Uses Retrieval-Augmented Generation (RAG) with FAISS + LangChain
- Download prediction results, training data, and CSV schema templates

---

## 🧱 Tech Stack

`Python`, `Streamlit`, `scikit-learn`, `XGBoost`, `FAISS`, `LangChain`, `Hugging Face`, `Ollama`, `LLaMA 3`

---

## 🛠️ Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/SuchetSanjeev/Supply_chain_Risk_Assessment-quantity-shortfall-Prediction-With-Smart-Query-Interface-.git
cd Supply_chain_Risk_Assessment-quantity-shortfall-Prediction-With-Smart-Query-Interface-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run main_streamlit.py
```

