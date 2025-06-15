import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

# Title
st.title("Enhanced Diabetes Risk Prediction Dashboard")

# Load dataset
data = pd.read_csv("data/diabetes.csv")

# Tabs for clean layout
tab1, tab2, tab3 = st.tabs(["Dataset", "Filtering", "Model & Evaluation"])

# --- Dataset tab ---
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

# --- Filtering tab ---
with tab2:
    age_min, age_max = int(data["Age"].min()), int(data["Age"].max())
    age_range = st.slider("Select Age Range:", age_min, age_max, (age_min, age_max))

    bmi_min, bmi_max = float(data["BMI"].min()), float(data["BMI"].max())
    bmi_range = st.slider("Select BMI Range:", bmi_min, bmi_max, (bmi_min, bmi_max))

    filtered_data = data[
        (data["Age"] >= age_range[0]) & (data["Age"] <= age_range[1]) &
        (data["BMI"] >= bmi_range[0]) & (data["BMI"] <= bmi_range[1])
    ]

    st.subheader(f"Data filtered by Age {age_range} and BMI {bmi_range}")
    st.dataframe(filtered_data)

# --- Model & Evaluation tab ---
with tab3:
    if not filtered_data.empty:
        X = filtered_data.drop("Outcome", axis=1)
        y = filtered_data["Outcome"]

        # Train models
        log_model = LogisticRegression(max_iter=1000)
        rf_model = RandomForestClassifier(n_estimators=100)

        log_model.fit(X, y)
        rf_model.fit(X, y)

        # Predictions
        log_preds = log_model.predict(X)
        rf_preds = rf_model.predict(X)

        # Accuracy
        log_acc = accuracy_score(y, log_preds)
        rf_acc = accuracy_score(y, rf_preds)

        st.subheader("Model Accuracy")
        st.write(f"Logistic Regression Accuracy: {log_acc:.2f}")
        st.write(f"Random Forest Accuracy: {rf_acc:.2f}")

        # Confusion matrices
        st.subheader("Confusion Matrices")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Logistic Regression")
            st.dataframe(pd.DataFrame(confusion_matrix(y, log_preds),
                                      columns=["Pred 0", "Pred 1"],
                                      index=["Actual 0", "Actual 1"]))

        with col2:
            st.write("Random Forest")
            st.dataframe(pd.DataFrame(confusion_matrix(y, rf_preds),
                                      columns=["Pred 0", "Pred 1"],
                                      index=["Actual 0", "Actual 1"]))

        # ROC curves
        st.subheader("ROC Curve Comparison")
        fig, ax = plt.subplots()

        for model_name, model in [("Logistic Regression", log_model), ("Random Forest", rf_model)]:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y, y_prob)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)

    else:
        st.warning("No data available in selected filter range!")





