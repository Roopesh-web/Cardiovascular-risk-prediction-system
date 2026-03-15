"""import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Set Page Config
st.set_page_config(page_title="Heart Diagnosis AI", layout="wide")

# Synthetic dataset
np.random.seed(42)
n_samples = 100

df = pd.DataFrame({
    "bp_last_week": np.random.randint(110, 160, n_samples),
    "chol_last_week": np.random.randint(170, 250, n_samples),
    "heart_rate_last_week": np.random.randint(65, 100, n_samples),
    "bp_this_week": np.random.randint(110, 160, n_samples),
    "chol_this_week": np.random.randint(170, 250, n_samples),
    "heart_rate_this_week": np.random.randint(65, 100, n_samples),
})

# Label: improved if this_week values are lower
df["condition"] = ((df["bp_this_week"] < df["bp_last_week"]) &
                   (df["chol_this_week"] < df["chol_last_week"]) &
                   (df["heart_rate_this_week"] <= df["heart_rate_last_week"])).astype(int)


# Convert DataFrame to NumPy Arrays
X_np = df.iloc[:, :-1].values
y_np = df['condition'].values

# Standardize Data for Better Performance
scaler = StandardScaler()
X_np = scaler.fit_transform(X_np)

# Train sklearn Logistic Regression Model
sklearn_model = LogisticRegression()
sklearn_model.fit(X_np, y_np)


st.sidebar.write("✅ 1 = Improved, ❌ 0 = Not Improved**")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")




# Main Title
st.title("💖 AI-Powered Heart Diagnosis")

# Two Columns for Input
col1, col2 = st.columns(2)

with col1:
    st.header("📅 Last Week's Health Data")
    bp_last_week = st.number_input("🩸 Blood Pressure (BP)", 50, 200, step=1, key="bp_last")
    chol_last_week = st.number_input("🧬 Cholesterol Level", 50, 300, step=1, key="chol_last")
    hr_last_week = st.number_input("❤ Heart Rate (HR)", 40, 150, step=1, key="hr_last")

with col2:
    st.header("📅 This Week's Health Data")
    bp_this_week = st.number_input("🩸 Blood Pressure (BP)", 50, 200, step=1, key="bp_this")
    chol_this_week = st.number_input("🧬 Cholesterol Level", 50, 300, step=1, key="chol_this")
    hr_this_week = st.number_input("❤ Heart Rate (HR)", 40, 150, step=1, key="hr_this")

# Prediction Button
if st.button("🔍 Analyze Health Condition"):
    input_data = [bp_last_week, chol_last_week, hr_last_week, bp_this_week, chol_this_week, hr_this_week]

    # Check if all fields are filled
    if any(x is None for x in input_data):
        st.error("⚠ Please fill all input fields!")
    else:
        with st.spinner("Analyzing your health..."):
            time.sleep(2)  # just to simulate processing

        input_array = np.array([input_data])
        input_scaled = scaler.transform(input_array)

        probabilities = sklearn_model.predict_proba(input_scaled)[0]
        prediction = 1 if probabilities[1] > 0.6 else 0

        # Results
        st.subheader("📊 Analysis Results")

        colA, colB = st.columns(2)
        with colA:
            st.metric("🩸 Blood Pressure Change", f"{bp_this_week - bp_last_week}", 
                      f"{(bp_this_week - bp_last_week)/bp_last_week:.1%}")
            st.metric("🧬 Cholesterol Change", f"{chol_this_week - chol_last_week}", 
                      f"{(chol_this_week - chol_last_week)/chol_last_week:.1%}")
            st.metric("❤ Heart Rate Change", f"{hr_this_week - hr_last_week}", 
                      f"{(hr_this_week - hr_last_week)/hr_last_week:.1%}")
        with colB:
            if prediction == 1:
                st.success("✅ Your condition has *IMPROVED*! 🎉")
                st.progress(probabilities[1])
                st.write(f"Confidence: *{probabilities[1]*100:.2f}%*")
                st.info("### Tips to Maintain Good Health:\n✔ Balanced diet 🥗\n✔ Regular exercise 🏃\n✔ Stay hydrated 💧\n✔ Manage stress 🧘")
            else:
                st.error("❌ Your condition has *NOT IMPROVED*. Please consult a doctor 🚨")
                st.progress(probabilities[0])
                st.write(f"Confidence: *{probabilities[0]*100:.2f}%*")
                st.warning("### Suggestions for Improvement:\n🔹 Reduce salt & sugar intake 🍬\n🔹 Increase physical activity 🚴\n🔹 Follow a heart-friendly diet 🥦\n🔹 Monitor BP & cholesterol regularly 🏥")

    #Visualization
    st.subheader("📈 Health Comparison Chart")
    categories = ["BP", "Cholesterol", "Heart Rate"]
    last_week = [bp_last_week, chol_last_week, hr_last_week]
    this_week = [bp_this_week, chol_this_week, hr_this_week]

    fig, ax = plt.subplots()
    x = np.arange(len(categories))
    ax.bar(x - 0.2, last_week, 0.4, label="Last Week")
    ax.bar(x + 0.2, this_week, 0.4, label="This Week")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Developed with ❤ for better heart health.")"""

import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, recall_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Heart Disease Prediction AI", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("HeartDiseaseTrain-Test.csv")
    df = df.dropna()
    return df

try:
    df = load_data()
except:
    st.error("❌ Dataset not found. Keep HeartDiseaseTrain-Test.csv in same folder.")
    st.stop()

# ---------------- SPLIT DATA ----------------
X = df.drop("target", axis=1)
y = df["target"]

categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

# ---------------- PREPROCESSOR ----------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
    ]
)

# ---------------- REALISTIC MODEL ----------------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    ))
])

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- TRAIN MODEL ----------------
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# ---------------- SIDEBAR ----------------
st.sidebar.title("ℹ Model Information")
st.sidebar.write("XGBoost model with SMOTE (Balanced & Regularized)")
st.sidebar.write(f"### Accuracy: {accuracy*100:.2f}%")
st.sidebar.write(f"### Recall (Disease Detection): {recall*100:.2f}%")
st.sidebar.write("---")
st.sidebar.write("Developed by Roopesh Sharma")

# ---------------- MAIN TITLE ----------------
st.title("💖 AI-Driven Cardiovascular Risk Assessment System")
st.subheader("🩺 Enter Patient Details")

input_data = {}
col1, col2 = st.columns(2)

numeric_half = len(numeric_cols) // 2

def create_integer_input(column):
    display_name = "ST Depression (Exercise ECG)" if column.lower() == "oldpeak" else column

    value = st.number_input(
        display_name,
        min_value=int(float(df[column].min())),
        max_value=int(float(df[column].max())),
        value=int(float(df[column].mean())),
        step=1,
        format="%d"
    )

    return int(value)

with col1:
    for column in numeric_cols[:numeric_half]:
        input_data[column] = create_integer_input(column)

with col2:
    for column in numeric_cols[numeric_half:]:
        input_data[column] = create_integer_input(column)

for column in categorical_cols:
    input_data[column] = st.selectbox(column, df[column].unique())

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Risk"):

    input_df = pd.DataFrame([input_data])

    with st.spinner("Analyzing patient data..."):
        time.sleep(2)

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.metric("Confidence Level", f"{probability*100:.2f}%")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with XGBoost, SMOTE & Proper Validation Strategy")