import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ==========================
# ⚙️ Helper Functions
# ==========================
def clean_column_names(df_):
    df_.columns = (
        df_.columns.str.replace(" ", "_")
        .str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
    )
    return df_

def preprocess_data(df_raw, le_type=None, fit_le=False):
    df = df_raw.copy()
    df = clean_column_names(df)
    df = df.drop(["UDI", "Product_ID"], axis=1, errors="ignore")

    current_le = le_type
    if "Type" in df.columns:
        if fit_le:
            current_le = LabelEncoder()
            df["Type_enc"] = current_le.fit_transform(df["Type"])
        elif current_le:
            df["Type_enc"] = df["Type"].apply(lambda x: current_le.transform([x])[0] if x in current_le.classes_ else -1)
        df = df.drop("Type", axis=1)

    all_failure_types = [
        "Failure_Type_No_Failure",
        "Failure_Type_Overstrain_Failure",
        "Failure_Type_Heat_Dissipation_Failure",
        "Failure_Type_Power_Failure",
        "Failure_Type_Random_Failures",
        "Failure_Type_Tool_Wear_Failure"
    ]

    for col in all_failure_types:
        if col not in df.columns:
            df[col] = 0

    for col in all_failure_types:
        df[col] = df[col].apply(lambda x: 0 if str(x).strip().upper() in ["NO FAILURE", "FALSE", "0", ""] else 1)
        df[col] = df[col].astype(int)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: 0 if str(x).strip().upper() in ["NO FAILURE", "FALSE", "0", ""] else 1)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if fit_le:
        return df, current_le
    return df, None

def forecast_failures(df_, model, le_type, scaler, forecast_days=10):
    df_new, _ = preprocess_data(df_, le_type=le_type, fit_le=False)

    feature_cols = [c for c in df_new.columns if c not in ["Timestamp", "Target"]]
    X_new_raw = df_new[feature_cols]

    X_new_scaled = scaler.transform(X_new_raw)
    X_new_df_scaled = pd.DataFrame(X_new_scaled, columns=feature_cols)

    df_new["Failure_Prob"] = model.predict_proba(X_new_df_scaled)[:, 1]
    df_new["Predicted_Failure"] = (df_new["Failure_Prob"] >= 0.5).astype(int)

    df_new["Forecast_Date"] = np.random.choice(
        [datetime.now().date() + timedelta(days=i) for i in range(forecast_days)],
        size=len(df_new),
        replace=True,
    )

    upcoming_failures = df_new[df_new["Predicted_Failure"] == 1]
    daily_summary = upcoming_failures.groupby("Forecast_Date").size().reset_index(name="Machines_Predicted_to_Fail")
    total_failures = daily_summary["Machines_Predicted_to_Fail"].sum()

    return daily_summary, total_failures

# ==========================
# 📦 Streamlit App
# ==========================
def main():
    st.set_page_config(layout="wide")
    st.title("🔮 Predictive Maintenance Failure Forecast Dashboard")
    st.markdown("Use this dashboard to view machine failure predictions and 10-day forecast summaries.")

    uploaded_file = st.file_uploader("📂 Upload your dataset (Excel)", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.success("✅ Dataset loaded successfully!")

        st.subheader("📊 Uploaded Dataset Preview")
        st.dataframe(df.head(5))

        with st.spinner("Processing data and training model..."):
            df_proc, le_type = preprocess_data(df, fit_le=True)

        st.subheader("🛠️ Processed Dataset Preview")
        st.dataframe(df_proc.head(5))

        if "Target" not in df_proc.columns:
            st.warning("⚠️ 'Target' column not found. Cannot train model.")
            return

        X = df_proc.drop(["Target", "Timestamp"], axis=1, errors="ignore")
        y = df_proc["Target"]

        X_numeric = X.select_dtypes(include=[np.number])

        if y.sum() == 0:
            st.error("❌ No positive failure cases in 'Target'. Cannot train model.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, stratify=y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_bal, y_train_bal)

        y_proba = rf.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        st.subheader("📈 Model Evaluation Metrics")
        col1, col2 = st.columns(2)
        col1.metric("ROC AUC", f"{roc_auc:.4f}")
        col2.metric("PR AUC", f"{pr_auc:.4f}")

        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred, digits=4))

        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # Forecast
        daily_summary, total_failures = forecast_failures(df, rf, le_type, scaler, forecast_days=10)

        st.subheader("🔮 10-Day Failure Forecast Summary")
        st.dataframe(daily_summary)

        st.markdown(f"### 🚨 Total Machines Predicted to Fail in Next 10 Days: **{total_failures}**")

        if not daily_summary.empty:
            daily_summary = daily_summary.sort_values("Forecast_Date")
            norm = plt.Normalize(daily_summary["Machines_Predicted_to_Fail"].min(),
                                 daily_summary["Machines_Predicted_to_Fail"].max())
            colors = [plt.cm.Reds(norm(val)) for val in daily_summary["Machines_Predicted_to_Fail"]]

            fig2, ax2 = plt.subplots(figsize=(8,4))
            sns.barplot(x="Forecast_Date", y="Machines_Predicted_to_Fail", data=daily_summary, ax=ax2, palette=colors)
            plt.xticks(rotation=45)
            ax2.set_title("Daily Machine Failure Forecast")
            st.pyplot(fig2)
        else:
            st.info("No failures predicted in the next 10 days.")

if __name__ == "__main__":
    main()
