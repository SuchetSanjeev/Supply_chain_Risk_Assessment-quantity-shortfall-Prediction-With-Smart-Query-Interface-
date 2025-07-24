import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from supply_chain_model import (
    feature_engineering, drop_unnecessary_columns, encode_features,
    split_and_scale_data, apply_smote, train_and_evaluate_model
)
# LangChain (RAG-related)
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOllama
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader

# === PAGE CONFIG ===
st.set_page_config(page_title="📦 Supply Chain Dashboard", layout="wide")

st.sidebar.title("📚 Navigation")
step = st.sidebar.radio("Go to", [
    "part 1 - Upload Files and Validate files",
    "part 2 - Merge Data and Train Model",
    "part 3 - Single Prediction",
    "part 4 - Batch Prediction",
])
page = st.sidebar.radio("Go to", ["📦 Prediction Interface", "RAG Query Interface"])

# === Page 1 ===
if page == "📦 Prediction Interface":
    st.title("📦 Supply Chain Shortfall Prediction Pipeline")
    # --- Expected Column Templates ---
    orders_expected_cols = [
        'Order_ID', 'Component_ID', 'Vendor_ID', 'Route_ID', 'Order_Date',
        'Contractual_Delivery_Date', 'Actual_Delivery_Date', 'Delivered_Qty',
        'Ordered_Qty', 'Price_per_Unit', 'Actual_Lead_Days', 'Committed_Lead_Days',
        'Shortfall', 'Shortfall_flag', 'Delay_Indicator'
    ]

    vendors_expected_cols = [
        'vendor_id', 'vendor_name', 'Reliability_score', 'avg_lead_days',
        'past_incident_count', 'Quality_Rejection_Rate (%)',
        'Supplier_Dependency_Index', 'collaboration_tenure'
    ]

    routes_expected_cols = [
        'Route_ID', 'Source', 'Destination', 'Mode', 'Avg_Transit_Days',
        'Weather_Disruption_Index', 'Route Risk Score',
        'Backup Route Availability', 'Peak_Congestion_Indicator'
    ]

    # --- Session State for DataFrames ---
    if 'orders_df' not in st.session_state: st.session_state['orders_df'] = None
    if 'vendors_df' not in st.session_state: st.session_state['vendors_df'] = None
    if 'routes_df' not in st.session_state: st.session_state['routes_df'] = None
    if 'merged_df' not in st.session_state: st.session_state['merged_df'] = None

    if step == "part 1 - Upload Files and Validate files":
        # === Step 1: Upload CSV Files ===
        st.header("📥 Step 1: Upload CSV Files")
        orders_file = st.file_uploader("Upload Orders CSV", type="csv")
        vendors_file = st.file_uploader("Upload Vendors CSV", type="csv")
        routes_file = st.file_uploader("Upload Routes CSV", type="csv")

        if orders_file:
            st.session_state.orders_df = pd.read_csv(orders_file)
            # st.success("✅ Orders file uploaded.")
        if vendors_file:
            st.session_state.vendors_df = pd.read_csv(vendors_file)
            # st.success("✅ Vendors file uploaded.")
        if routes_file:
            st.session_state.routes_df = pd.read_csv(routes_file)
            # st.success("✅ Routes file uploaded.")

        # === Step 2: Validate Columns ===
        st.header("📋 Step 2: Validate Uploaded Files")

        def check_columns(uploaded_df, expected_cols, label):
            missing = [col for col in expected_cols if col not in uploaded_df.columns]
            extra = [col for col in uploaded_df.columns if col not in expected_cols]
            if missing:
                st.error(f"❌ {label} is missing: {missing}")
            else:
                st.success(f"✅ {label} has all required columns.")
            if extra:
                st.info(f"ℹ️ {label} has extra columns: {extra}")

        if st.session_state.orders_df is not None:
            check_columns(st.session_state.orders_df, orders_expected_cols, "Orders")

        if st.session_state.vendors_df is not None:
            check_columns(st.session_state.vendors_df, vendors_expected_cols, "Vendors")

        if st.session_state.routes_df is not None:
            check_columns(st.session_state.routes_df, routes_expected_cols, "Routes")

    if step == "part 2 - Merge Data and Train Model":
        #THIS IS THE MERGING THAT I USED TO DO BUT I CHANGES ACCORDING TO THE MAIL SENT
        # === Step 3: Merge Data ===
        # st.header("🔗 Step 3: Merging the Uploaded files")

        # if st.button("🔄 Merge Now"):
        #     try:
        #         merged = pd.merge(st.session_state.orders_df, st.session_state.vendors_df,
        #                         left_on="Vendor_ID", right_on="vendor_id", how="left")
        #         merged = pd.merge(merged, st.session_state.routes_df, on="Route_ID", how="left")
        #         st.session_state.merged_df = merged
        #         st.success("✅ Merged Dataset created successfully!")
        #         st.dataframe(merged.head())
        #     except Exception as e:
        #         st.error(f"❌ Merge failed: {e}")

        # === Step 3: Merge Data (Case-Insensitive) ===
        st.header("🔗 Step 3: Merge Uploaded Files")

        orders_df = st.session_state.get("orders_df")
        vendors_df = st.session_state.get("vendors_df")
        routes_df = st.session_state.get("routes_df")

        def get_common_columns_case_insensitive(df1, df2):
            df1_cols_lower = {col.lower(): col for col in df1.columns}
            df2_cols_lower = {col.lower(): col for col in df2.columns}
            common_keys_lower = set(df1_cols_lower.keys()) & set(df2_cols_lower.keys())
            return [(df1_cols_lower[k], df2_cols_lower[k]) for k in common_keys_lower]

        # === Step 3.1 - Merge Orders and Vendors ===
        if orders_df is not None and vendors_df is not None:
            st.subheader("Step 3.1 - Merge Orders and Vendors")

            common_cols_ov = get_common_columns_case_insensitive(orders_df, vendors_df)
            if common_cols_ov:
                display_names_ov = [f"{left} ↔ {right}" for left, right in common_cols_ov]
                selected_pair_ov = st.selectbox("Select common column to merge Orders ↔ Vendors", display_names_ov)

                if st.button("🔗 Merge Orders with Vendors"):
                    try:
                        left_key, right_key = selected_pair_ov.split(" ↔ ")
                        merged_ov = pd.merge(
                            orders_df, vendors_df,
                            left_on=left_key.strip(), right_on=right_key.strip(), how="left"
                        )
                        st.session_state.merged_ov = merged_ov
                        st.success("✅ Orders and Vendors merged successfully!")
                        st.dataframe(merged_ov.head())
                    except Exception as e:
                        st.error(f"❌ Merge Orders ↔ Vendors failed: {e}")
            else:
                st.warning("⚠️ No common columns between Orders and Vendors to merge.")

        # === Step 3.2 - Merge Above with Routes ===
        if st.session_state.get("merged_ov") is not None and routes_df is not None:
            st.subheader("Step 3.2 - Merge Above with Routes")
            merged_ov = st.session_state.merged_ov

            common_cols_r = get_common_columns_case_insensitive(merged_ov, routes_df)
            if common_cols_r:
                display_names_r = [f"{left} ↔ {right}" for left, right in common_cols_r]
                selected_pair_r = st.selectbox("Select common column to merge with Routes", display_names_r)

                if st.button("🔗 Final Merge with Routes"):
                    try:
                        left_key, right_key = selected_pair_r.split(" ↔ ")
                        final_merged = pd.merge(
                            merged_ov, routes_df,
                            left_on=left_key.strip(), right_on=right_key.strip(), how="left"
                        )
                        st.session_state.merged_df = final_merged
                        st.success("✅ Final merged dataset (Orders + Vendors + Routes) created!")
                        st.dataframe(final_merged.head())
                    except Exception as e:
                        st.error(f"❌ Merge with Routes failed: {e}")
            else:
                st.warning("⚠️ No common columns between merged data and Routes to merge.")


        # # === Step 4: Train Model ===
        # st.header("🧠 Step 4: Train a Model")

        # if st.session_state.merged_df is not None:
        #     if st.button("🚀 Train Model"):
        #         with st.spinner("Training..."):
        #             df = feature_engineering(st.session_state.merged_df)
        #             target = df['Shortfall_flag'].copy()
        #             df = drop_unnecessary_columns(df)
        #             df_encoded = encode_features(df)

        #             X = df_encoded.copy()
        #             y = target
        #             X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
        #             X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)

        #             model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        #             train_and_evaluate_model(model, "Random Forest", X_train_res, y_train_res, X_test_scaled, y_test)

        #             joblib.dump(model, "model2.pkl")
        #             joblib.dump(scaler, "scaler2.pkl")
        #             joblib.dump(X.columns.tolist(), "columns2.pkl")
        #             st.success("Trained model saved.")

        #             training_output_df = df.copy()
        #             training_output_df['Shortfall_flag'] = target
        #             training_output_df.to_csv("knowledge_training_data.csv", index=False)
        #             # st.info("💾 Saved `knowledge_training_data.csv` for RAG-based querying.")

        #             y_probs = model.predict_proba(X_test_scaled)[:, 1]
        #             y_preds = (y_probs >= 0.5).astype(int)
        #             result_df = pd.DataFrame(X_test_scaled, columns=X.columns)
        #             result_df['True_Shortfall'] = y_test.reset_index(drop=True)
        #             result_df['Predicted_Shortfall'] = y_preds
        #             result_df['Prediction_Probability'] = y_probs

        #             csv = result_df.to_csv(index=False).encode()
        #             st.download_button(
        #                 label="📥 Download Test Predictions",
        #                 data=csv,
        #                 file_name="shortfall_test_predictions.csv",
        #                 mime="text/csv"
        #             )
        
        # === Step 4: Train Model ===
        st.header("🧠 Step 4: Train a Model")
        if st.session_state.get("merged_df") is not None:
            model_choice = st.selectbox("Choose a model to train", ["Random Forest", "XGBoost", "Logistic Regression", "SVM"])

            if st.button("🚀 Train Selected Model"):
                with st.spinner("Training..."):
                    df = feature_engineering(st.session_state.merged_df)
                    target = df['Shortfall_flag'].copy()
                    df = drop_unnecessary_columns(df)
                    df_encoded = encode_features(df)

                    X = df_encoded.copy()
                    y = target
                    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
                    X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)

                    # === Select and train model ===
                    if model_choice == "Random Forest":
                        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
                    elif model_choice == "XGBoost":
                        model = XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=6,reg_alpha=0.5)
                    elif model_choice == "Logistic Regression":
                        model = LogisticRegression(penalty='l1',solver='liblinear',C=1.0,random_state=42)
                    elif model_choice == "SVM":
                        model = SVC(probability=True, kernel='rbf')

                    results = train_and_evaluate_model(model, model_choice, X_train_res, y_train_res, X_test_scaled, y_test)
                    st.subheader("📊 Model Evaluation Metrics")
                    st.write(f"*Accuracy:* {results['accuracy']:.4f}")
                    st.subheader("📉 Classification Report")
                    st.json(results['report'])
                    st.subheader("📌 Confusion Matrix")
                    st.dataframe(results['confusion_matrix'])

                    # Save artifacts
                    joblib.dump(model, "model2.pkl")
                    joblib.dump(scaler, "scaler2.pkl")
                    joblib.dump(X.columns.tolist(), "columns2.pkl")
                    st.success(f"{model_choice} model trained and saved successfully.")

                    # Save knowledge base
                    training_output_df = df.copy()
                    training_output_df['Shortfall_flag'] = target
                    training_output_df.to_csv("knowledge_training_data.csv", index=False)

                    # Export predictions
                    y_probs = model.predict_proba(X_test_scaled)[:, 1]
                    y_preds = (y_probs >= 0.5).astype(int)
                    result_df = pd.DataFrame(X_test_scaled, columns=X.columns)
                    result_df['True_Shortfall'] = y_test.reset_index(drop=True)
                    result_df['Predicted_Shortfall'] = y_preds
                    result_df['Prediction_Probability'] = y_probs

                    csv = result_df.to_csv(index=False).encode()
                    st.download_button(
                        label="📥 Download Test Predictions",
                        data=csv,
                        file_name="shortfall_test_predictions.csv",
                        mime="text/csv"
                    )

                 
    if step == "part 3 - Single Prediction":
        # === Step 5: Predict for New Orders ===
        st.header("📈 Step 5: Custom Input prediction")

        def load_prediction_artifacts():
            model = joblib.load("model2.pkl")
            scaler = joblib.load("scaler2.pkl")
            columns = joblib.load("columns2.pkl")
            return model, scaler, columns

        def predict_shortfall_from_input(input_data):
            model, scaler, reference_columns = load_prediction_artifacts()
            df = pd.DataFrame([input_data])
            df = feature_engineering(df)
            df = drop_unnecessary_columns(df)
            df = encode_features(df)
            df = df.reindex(columns=reference_columns, fill_value=0)
            df_scaled = scaler.transform(df)
            prob = model.predict_proba(df_scaled)[0][1]
            flag = int(prob >= 0.5)
            return prob, flag

        with st.form("shortfall_form"):
            st.subheader("📝 New Order Input Form")

            col1, col2, col3 = st.columns(3)
            with col1:
                Ordered_Qty = st.number_input("Ordered Qty", value=100)
                Committed_Lead_Days = st.number_input("Committed Lead Days", value=30)
                Reliability_score = st.slider("Reliability Score", 0, 100, 80)
                Quality_Rejection = st.slider("Quality Rejection Rate (%)", 0, 100, 5)
                collaboration_tenure = st.slider("Collaboration Tenure (years)", 0, 10, 2)

            with col2:
                avg_lead_days = st.number_input("Average Lead Days", value=7)
                past_incident_count = st.number_input("Past Incident Count", value=1)
                Price_per_Unit = st.number_input("Price per Unit", value=60.0)
                Mode = st.selectbox("Transport Mode", ['Air', 'Lorry', 'Train'])
                Peak_Congestion = st.selectbox("Peak Congestion Indicator", ['Low', 'Medium', 'High'])

            with col3:
                Backup_Availability = st.selectbox("Backup Route Availability", ['Yes', 'No'])
                Avg_Transit_Days = st.number_input("Avg Transit Days", value=3)
                Weather_Index = st.slider("Weather Disruption Index", 0.0, 10.0, 1.0)
                Route_Risk = st.slider("Route Risk Score", 0.0, 10.0, 2.0)
                Component_ID = st.number_input("Component ID", value=1)
                Vendor_ID = st.number_input("Vendor ID", value=5)
                Route_ID = st.number_input("Route ID", value=6)
                Source = st.number_input("Source", value=12)

            submitted = st.form_submit_button("📤 Predict Shortfall")

            if submitted:
                input_dict = {
                    'Ordered_Qty': Ordered_Qty,
                    'Committed_Lead_Days': Committed_Lead_Days,
                    'Reliability_score': Reliability_score,
                    'Quality_Rejection_Rate (%)': Quality_Rejection,
                    'collaboration_tenure': collaboration_tenure,
                    'avg_lead_days': avg_lead_days,
                    'past_incident_count': past_incident_count,
                    'Price_per_Unit': Price_per_Unit,
                    'Mode': Mode,
                    'Peak_Congestion_Indicator': Peak_Congestion,
                    'Backup Route Availability': Backup_Availability,
                    'Avg_Transit_Days': Avg_Transit_Days,
                    'Weather_Disruption_Index': Weather_Index,
                    'Route Risk Score': Route_Risk,
                    'Component_ID': Component_ID,
                    'Vendor_ID': Vendor_ID,
                    'Route_ID': Route_ID,
                    'Source': Source
                }

                prob, flag = predict_shortfall_from_input(input_dict)

                st.success("✅ Prediction Complete")
                st.write(f"**Shortfall Probability:** `{prob:.4f}`")
                st.write(f"**Predicted Shortfall Flag:** `{'Yes' if flag else 'No'}`")

    # if step == "part 4 - Batch Prediction":
    #     # === Step 6: Batch Prediction ===
    #     st.header("📄 Step 6: Batch Prediction for Uploaded Order File")
    #     batch_file = st.file_uploader("Upload New Orders CSV", type="csv", key="batch")

    #     def batch_predict(df_batch):
    #         model = joblib.load("model2.pkl")
    #         scaler = joblib.load("scaler2.pkl")
    #         reference_columns = joblib.load("columns2.pkl")

    #         original_df = df_batch.copy()
    #         df_batch = feature_engineering(df_batch)
    #         df_batch = drop_unnecessary_columns(df_batch)
    #         df_batch = encode_features(df_batch)
    #         df_batch = df_batch.reindex(columns=reference_columns, fill_value=0)
    #         df_scaled = scaler.transform(df_batch)

    #         probs = model.predict_proba(df_scaled)[:, 1]
    #         flags = (probs >= 0.5).astype(int)

    #         original_df['Shortfall_Probability'] = probs
    #         original_df['Predicted_Shortfall_Flag'] = flags
    #         return original_df

    #     if batch_file:
    #         batch_df = pd.read_csv(batch_file)
    #         st.markdown("📋 Preview of Uploaded File:")
    #         st.dataframe(batch_df.head())

    #         if st.button("🔍 Predict Shortfalls for Uploaded Orders"):
    #             with st.spinner("Predicting..."):
    #                 results = batch_predict(batch_df)
    #                 st.success("✅ Batch Predictions Complete!")
    #                 st.dataframe(results.head())

    #                 csv_data = results.to_csv(index=False).encode()
    #                 st.download_button(
    #                     label="📥 Download Batch Predictions CSV",
    #                     data=csv_data,
    #                     file_name="batch_predictions.csv",
    #                     mime="text/csv"
    #                 )

    #                 results.to_csv("knowledge_batch_predictions.csv", index=False)
    #                 # st.info("💾 Saved `knowledge_batch_predictions.csv` for future RAG-based Q&A.")
    
    if step == "part 4 - Batch Prediction":
    # === Step 6: Batch Prediction ===
        st.header("📄 Step 6: Batch Prediction for Uploaded Order File")

        # --- Add Template Download Button ---
        st.subheader("📤 Download Input Template")
        if st.button("📄 Download Template CSV for Batch Prediction"):
            template_columns = [
                'Ordered_Qty', 'Committed_Lead_Days', 'Reliability_score',
                'Quality_Rejection_Rate (%)', 'collaboration_tenure', 'avg_lead_days',
                'past_incident_count', 'Price_per_Unit', 'Mode', 'Peak_Congestion_Indicator',
                'Backup Route Availability', 'Avg_Transit_Days', 'Weather_Disruption_Index',
                'Route Risk Score', 'Component_ID', 'Vendor_ID', 'Route_ID', 'Source'
            ]
            template_df = pd.DataFrame(columns=template_columns)
            csv_template = template_df.to_csv(index=False).encode()

            st.download_button(
                label="📥 Click to Download Template CSV",
                data=csv_template,
                file_name="batch_prediction_template.csv",
                mime="text/csv"
            )

        # --- Batch Upload & Prediction ---
        batch_file = st.file_uploader("Upload New Orders CSV", type="csv", key="batch")

        def batch_predict(df_batch):
            model = joblib.load("model2.pkl")
            scaler = joblib.load("scaler2.pkl")
            reference_columns = joblib.load("columns2.pkl")

            original_df = df_batch.copy()
            df_batch = feature_engineering(df_batch)
            df_batch = drop_unnecessary_columns(df_batch)
            df_batch = encode_features(df_batch)
            df_batch = df_batch.reindex(columns=reference_columns, fill_value=0)
            df_scaled = scaler.transform(df_batch)

            probs = model.predict_proba(df_scaled)[:, 1]
            flags = (probs >= 0.5).astype(int)

            original_df['Shortfall_Probability'] = probs
            original_df['Predicted_Shortfall_Flag'] = flags

            return original_df

        if batch_file:
            batch_df = pd.read_csv(batch_file)
            st.markdown("📋 Preview of Uploaded File:")
            st.dataframe(batch_df.head())

            if st.button("🔍 Predict Shortfalls for Uploaded Orders"):
                with st.spinner("Predicting..."):
                    results = batch_predict(batch_df)

                    st.success("✅ Batch Predictions Complete!")
                    st.dataframe(results.head())

                    csv_data = results.to_csv(index=False).encode()
                    st.download_button(
                        label="📥 Download Batch Predictions CSV",
                        data=csv_data,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )

                    results.to_csv("knowledge_batch_predictions.csv", index=False)
                    # st.info("💾 Saved `knowledge_batch_predictions.csv` for future RAG-based Q&A.")


# === Page 2 ===
elif page == "RAG Query Interface":
    st.title("💬 Step 7: Ask Questions to Your Supply Chain Model (RAG + LLaMA 3)")
    st.markdown("""
    Use natural language to ask questions about:
    - Vendor performance
    - Route risk
    - Shortfall causes
    - Congestion impact
    - Model predictions
    """)

    @st.cache_resource
    def load_qa_chain():
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            llm = ChatOllama(model="llama3:8b", temperature=0.2,stream=True)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
            # st.success("✅ FAISS index and LLaMA 3 initialized successfully!")
            return qa_chain
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {e}")
            return None

    qa_chain = load_qa_chain()

    if qa_chain:
        user_query = st.text_input("💬 Ask your question:")
        if user_query:
            with st.spinner("🔎 Thinking..."):
                try:
                    result = qa_chain(user_query)
                    st.markdown("### ✅ Answer")
                    st.write(result['result'])
                    with st.expander("📄 Sources"):
                        for doc in result['source_documents']:
                            st.markdown(f"`...{doc.page_content[:400]}...`")
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
