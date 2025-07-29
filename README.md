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

## 🖥 Run the Streamlit App

streamlit run streamlit_csv_validator.py

Navigate to http://localhost:8501 in your browser.


---

## 📦 Upload Guidelines

Orders CSV: Must contain columns like Order_ID, Component_ID, Ordered_Qty, etc.

Vendors CSV: Must include reliability scores, incident counts, etc.

Routes CSV: Should include transit days, risk score, mode, etc.


Supports fuzzy matching of column names.


---

## 💡 Example Queries for RAG

"Which vendor has the lowest incident count?"

"What are the top risks for Route 12?"

"Show delivery delays for March 2024."

"What are the conditions that affect Vendor X performance?"

"What is the recommended backup route strategy?"



---

## 📌 Dependencies

Key packages:

pandas, scikit-learn, xgboost, imblearn, joblib

streamlit, matplotlib, seaborn

langchain, langchain-community, fuzzywuzzy

ollama (external install)


---

## 🤝 License & Credits

Built with ❤ by Suchet Patil
AI Capstone Project | VIT Vellore

---

## 📬 Contact

Have suggestions or issues?
Raise an issue or mail: suchet.patil@example.com

---

# My understanding of the project overview

- So basically to run this : put these files Order_Data_Final/Order_Data_Final_for_fuzzer_test.csv , vendor_Data_Final/vendor_Data_Final_for_fuzzer_test.csv , Route_Data_Final/Route_Data_Final_for_fuzzer_test.csv into the empty file upload of the interface (1) where we upload csv then it will use fuzzers logic and match it with the predefined schema in the backend and then it will verify and will help me validate my columns in next step (2).
- After which we move onto the next step (3) which is merging the tables or uploaded csv's , firstly the merging is performed on order table and vendor table  based on the common columns, then my merging is performed on the currently merged dataset with the route table to give the final merged dataset which is used by the model for prediction in the backend
- Now in the next step (4) i use this dataset to train my model using any of the 4 classification algorithm(RF,LR,XGB,SVM) that has been displayed. 
- After this comes the testing part which is the next step (5) for custom input/single input form format adjusted inputs for quantity shortfall prediction.
- Then you could also perform bulk/batch prediction of unseen data in step 6 which takes here input file knowledge_batch_predictions_without_shortfall.csv as input and then we download the predictions and see the output. then our file knowledge_training_data is used to help us fine tune our LLM for the query interface (done in the backend). 
- Now in my final step (7) i could go to the Query interface where i ask any query related to my supply chain project built and it will help the customer try to resolve this issue.


