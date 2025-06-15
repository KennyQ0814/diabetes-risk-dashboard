import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Title
st.title("Diabetes Risk Prediction Dashboard")

# Load dataset
data = pd.read_csv("data/diabetes.csv")

# Preview
st.subheader("Dataset Preview")
st.dataframe(data.head())

# Age filter
age_min, age_max = int(data["Age"].min()), int(data["Age"].max())
age_range = st.slider("Select Age Range:", age_min, age_max, (age_min, age_max))
filtered_data = data[(data["Age"] >= age_range[0]) & (data["Age"] <= age_range[1])]

# BMI filter
bmi_min, bmi_max = float(data["BMI"].min()), float(data["BMI"].max())
bmi_range = st.slider("Select BMI Range:", bmi_min, bmi_max, (bmi_min, bmi_max))
filtered_data = filtered_data[(filtered_data["BMI"] >= bmi_range[0]) & (filtered_data["BMI"] <= bmi_range[1])]

# Show filtered data
st.subheader(f"Data filtered by Age {age_range} and BMI {bmi_range}")
st.dataframe(filtered_data)

# Model training
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Model prediction on filtered data
if not filtered_data.empty:
    preds = model.predict(filtered_data.drop("Outcome", axis=1))
    filtered_data = filtered_data.copy()
    filtered_data["Prediction"] = preds
    st.subheader("Filtered Data with Predictions")
    st.dataframe(filtered_data)

    # Visualize glucose levels
    st.subheader("Glucose Levels in Filtered Data")
    st.bar_chart(filtered_data["Glucose"])

    # Visualize prediction distribution
    st.subheader("Prediction Outcome Distribution (0 = no diabetes, 1 = diabetes)")
    st.bar_chart(filtered_data["Prediction"].value_counts())
else:
    st.warning("No data matches the selected filters. Adjust the sliders to see results.")




