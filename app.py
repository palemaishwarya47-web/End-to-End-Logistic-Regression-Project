import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    font-size: 42px;
    font-weight: bold;
    color: #1f3b73;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}

.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.metric-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE SECTION
# ---------------------------------------------------

st.markdown(
    '<p class="title">🎓 Student Performance Prediction App</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict Final Student Marks using Machine Learning</p>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("student_data_set.csv")

# ---------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------

df = df.fillna(df.mean(numeric_only=True))
df = df.drop_duplicates()

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------

df["Study_Efficiency"] = (
    df["Final_Marks"] / df["Hours_Studied"]
)

# ---------------------------------------------------
# FEATURES & TARGET
# ---------------------------------------------------

X = df.drop("Final_Marks", axis=1)
y = df["Final_Marks"]

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

model = LinearRegression()

model.fit(X_train_scaled, y_train)

# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------

pickle.dump(model, open("student_model.pkl", "wb"))

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("📌 About Project")

st.sidebar.info("""
This project predicts student final marks using:

✅ Linear Regression  
✅ Feature Scaling  
✅ Data Cleaning  
✅ Feature Engineering  

Built using:
- Python
- Streamlit
- Scikit-learn
- Pandas
""")

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

st.header("📥 Enter Student Details")

col1, col2, col3 = st.columns(3)

with col1:

    hours_studied = st.slider(
        "Hours Studied",
        1, 24, 5
    )

    attendance = st.slider(
        "Attendance (%)",
        0, 100, 75
    )

    assignments = st.slider(
        "Assignments Score",
        0, 100, 70
    )

with col2:

    sleep_hours = st.slider(
        "Sleep Hours",
        1, 12, 7
    )

    previous_marks = st.slider(
        "Previous Marks",
        0, 100, 65
    )

    internet_usage = st.slider(
        "Internet Usage Hours",
        0, 24, 4
    )

with col3:

    activities = st.slider(
        "Extracurricular Activities",
        0, 10, 2
    )

    mock_test = st.slider(
        "Mock Test Score",
        0, 10, 6
    )

    study_efficiency = st.slider(
        "Study Efficiency",
        1.0, 100.0, 10.0
    )

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

if st.button("🚀 Predict Final Marks"):

    new_student = [[
        hours_studied,
        attendance,
        assignments,
        sleep_hours,
        previous_marks,
        internet_usage,
        activities,
        mock_test,
        study_efficiency
    ]]

    # Scale data
    new_student_scaled = scaler.transform(new_student)

    # Prediction
    prediction = model.predict(new_student_scaled)

    predicted_marks = prediction[0]

    st.success(
        f"🎯 Predicted Final Marks: {predicted_marks:.2f}"
    )

    # Performance Message
    if predicted_marks >= 90:
        st.balloons()
        st.success("Excellent Performance 🌟")

    elif predicted_marks >= 75:
        st.info("Good Performance 👍")

    elif predicted_marks >= 50:
        st.warning("Average Performance 📘")

    else:
        st.error("Needs Improvement 📉")

# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------

st.header("📊 Dataset Preview")

st.dataframe(df.head())

# ---------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------

st.header("📈 Data Visualizations")

col4, col5 = st.columns(2)

with col4:

    fig1, ax1 = plt.subplots(figsize=(6,4))

    ax1.hist(df["Final_Marks"], bins=10)

    ax1.set_title("Distribution of Final Marks")
    ax1.set_xlabel("Final Marks")
    ax1.set_ylabel("Frequency")

    st.pyplot(fig1)

with col5:

    fig2, ax2 = plt.subplots(figsize=(6,4))

    ax2.scatter(
        df["Hours_Studied"],
        df["Final_Marks"]
    )

    ax2.set_title("Hours Studied vs Final Marks")
    ax2.set_xlabel("Hours Studied")
    ax2.set_ylabel("Final Marks")

    st.pyplot(fig2)

# ---------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------

st.header("🤖 Model Information")

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        label="Model Used",
        value="Linear Regression"
    )

with info2:
    st.metric(
        label="Dataset Rows",
        value=df.shape[0]
    )

with info3:
    st.metric(
        label="Features",
        value=X.shape[1]
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown(
    "<center>Made with ❤️ using Streamlit</center>",
    unsafe_allow_html=True
)