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

`Python`, `Streamlit`, `scikit-learn`, `FAISS`, `LangChain`, `Hugging Face`, `LLaMA 3`, `pandas`, `Numpy`, `matplotlib`, `seaborn`

---

## 🛠️ Setup Instructions

# 1. Clone the repository
```bash
git clone https://github.com/SuchetSanjeev/Supply_chain_Risk_Assessment-quantity-shortfall-Prediction-With-Smart-Query-Interface-.git
```

# 2. Install dependencies
```bash
pip install -r requirements.txt
```
# or you could also manually add inside requirements.txt

# 3. Run the Streamlit app
```bash
streamlit run main_streamlit.py
```

## 🤖 For Enabling the LLaMA 3 Chat Interface via Ollama

This project uses **LLaMA 3** locally via **Ollama** to power the Retrieval-Augmented Generation (RAG) question-answering interface.

### 📥 Step-by-Step Instructions

1. **Download and Install Ollama**

   Visit the official Ollama download page and install it for your OS:

   👉 [https://ollama.com/download](https://ollama.com/download)

2. *🧪 Check Ollama Installation*

   ```bash
    ollama --version
   ```
3. *Download LLaMa3 model(8B version)*
   ```bash
    ollama pull llama3:8b
   ```
4. *start ollama server*
   ```bash
    ollama serve  
   ```
5. *list available models*
   ```bash
    ollama list
   ```

---

# Working of my project

- So basically to run this : put Order_Data_Final , vendor_Data_Final csv into the blank the interface part 1 where we upload csv  then based on the predefined schema in the backend it will verify and will help me validate my columns in step 2.
- After which we move onto step 3 which is merging the tables or uploaded csv's , firstly the merging is performed on order table and vendor table  based on the common columns, then my merging is performed on the currently merged dataset with the route table to give the final merged dataset which is used by the model for prediction in the backend
- Now in step 4 i use this dataset to train my model using any of the 4 classification algorithm that has been displayed. 
- After this comes the testing part which is step 5 for custom input/single input form format adjusted inputs for quantity shortfall prediction.
- Then you could also perform bulk/batch prediction of unseen data in step 6 which takes here input file knowledge_batch_predictions_without_shortfall.csv as input and then we download the predictions and see the output. then our file knowledge_training_data is used to help us fine tune our LLM for the query interface (done in the backend). 
- Now in my final step 7 i could go to the Query interface where i ask any query related to my supply chain project built and it will help the customer try to resolve this issue.

---
