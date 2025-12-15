import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# =========================
# APP TITLE
# =========================
st.set_page_config(page_title="AI Career Recommendation", layout="centered")
st.title("🎓 AI-Based Career Recommendation System")

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("career_dataset.csv")

TARGET = "Career"

# =========================
# ENCODING (MODEL.PY LOGIC)
# =========================
encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# =========================
# SPLIT FEATURES & LABEL
# =========================
X = df.drop(TARGET, axis=1)
y = df[TARGET]

feature_names = X.columns.tolist()

# =========================
# TRAIN MODEL (INSIDE APP)
# =========================
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X, y)

# =========================
# USER INPUT FORM
# =========================
st.subheader("🧠 Enter Your Information")

user_input = []

for feature in feature_names:
    if feature in encoders:
        value = st.selectbox(
            feature,
            encoders[feature].classes_
        )
        encoded_value = encoders[feature].transform([value])[0]
        user_input.append(encoded_value)
    else:
        value = st.number_input(feature, min_value=0, max_value=100)
        user_input.append(value)

# =========================
# PREDICTION
# =========================
if st.button("🔍 Recommend Career"):
    prediction = model.predict([user_input])[0]
    career = encoders[TARGET].inverse_transform([prediction])[0]

    st.success(f"✅ Recommended Career: **{career}**")
