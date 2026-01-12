import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="ASD Screening Tool",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Autism Spectrum Disorder Screening and prediction")
st.caption("This is a screening tool, not a medical diagnosis.")
st.divider()

# ==================================================
# LOAD MODELS (CACHED)
# ==================================================
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("logistic_regression.pkl"),
        "Random Forest": joblib.load("random_forest.pkl"),
        "XGBoost": joblib.load("xgboost.pkl")
    }

models = load_models()

# ==================================================
# QUESTIONNAIRE
# ==================================================
questions = {
    "A1_Score": "Difficulty making new friends",
    "A2_Score": "Prefers to do things alone",
    "A3_Score": "Finds social situations confusing",
    "A4_Score": "Difficulty understanding others' feelings",
    "A5_Score": "Misses social cues",
    "A6_Score": "Strong focus on specific interests",
    "A7_Score": "Strong preference for routines",
    "A8_Score": "Avoids eye contact",
    "A9_Score": "Struggles with small talk",
    "A10_Score": "Finds group conversations difficult"
}

st.subheader("📋 Questionnaire for prediction")
responses = {k: st.radio(v, ["No", "Yes"], horizontal=True) for k, v in questions.items()}

# ==================================================
# DEMOGRAPHICS
# ==================================================
st.subheader("👤 Demographic Information")
age = st.number_input("Age", min_value=1, max_value=100, value=20)
gender = st.selectbox("Gender", ["Female", "Male"])
family_asd = st.selectbox("Family history of ASD?", ["No", "Yes"])

# ==================================================
# INPUT PREPARATION
# ==================================================
user_input = {k: 1 if responses[k] == "Yes" else 0 for k in responses}
user_input["age"] = age
user_input["gender"] = 1 if gender == "Male" else 0
user_input["family_mem_with_ASD"] = 1 if family_asd == "Yes" else 0

# ==================================================
# MODEL SELECTION
# ==================================================
st.subheader("🧪 Model Selection")
model_name = st.selectbox("Select a model:", list(models.keys()), index=2)
model = models[model_name]

# ==================================================
# FEATURE ALIGNMENT
# ==================================================
final_input = {f: user_input.get(f, 0) for f in model.feature_names_in_}
input_df = pd.DataFrame([final_input])

# ==================================================
# PREDICTION
# ==================================================
if st.button("🔍 Predict ASD Risk"):
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Screening Result")
    st.metric("ASD Risk Probability", f"{probability:.2f}")

    if probability >= 0.6:
        st.error("🔴 High likelihood of ASD traits. Clinical evaluation recommended.")
    elif probability >= 0.3:
        st.warning("🟠 Moderate likelihood of ASD traits. Further screening advised.")
    else:
        st.success("🟢 Low likelihood of ASD traits.")

    # ==================================================
    # SHAP EXPLANATION
    # ==================================================
    st.subheader("🧩 Explanation of Prediction")

    if model_name == "Logistic Regression":
        st.info("Logistic Regression explanation using model coefficients.")

        coef_df = pd.DataFrame({
            "Feature": input_df.columns,
            "Input Value": input_df.iloc[0],
            "Coefficient": model.coef_[0]
        }).sort_values(by="Coefficient", ascending=False)

        st.dataframe(coef_df, use_container_width=True)

    else:
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)

        # Robust handling for binary / multi-output SHAP
        if shap_values.values.ndim == 3:
            shap_vals = shap_values.values[0, :, 1]
            base_val = shap_values.base_values[0, 1]
        else:
            shap_vals = shap_values.values[0]
            base_val = shap_values.base_values[0]

        shap_exp = shap.Explanation(
            values=shap_vals,
            base_values=base_val,
            data=input_df.iloc[0],
            feature_names=input_df.columns
        )

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_exp, show=False)
        st.pyplot(fig)

    # ==================================================
    # RULE-BASED + SHAP-DRIVEN AGENT
    # ==================================================
    st.subheader("🧠 Decision-Support Agent")

    if probability >= 0.6:
        st.error("Strong recommendation: consult a developmental specialist.")
    elif probability >= 0.3:
        st.warning("Professional screening and monitoring advised.")
    else:
        st.success("Routine developmental monitoring is sufficient.")

st.divider()


