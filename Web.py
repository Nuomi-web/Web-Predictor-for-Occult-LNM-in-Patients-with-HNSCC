import streamlit as st
import joblib
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# ===============================
# 1. Load model
# ===============================
model_xgb = joblib.load('XGBoost.pkl')

# ===============================
# 2. Feature names shown on web page
# ===============================
feature_label = [
    'IC_DL_57',
    'VMI_original_glszm_SmallAreaHighGrayLevelEmphasis',
    'IC_wavelet-LLH_glszm_ZoneEntropy',
    'Zeff_DL_91',
    'VMI_DL_45',
    'IC_DL_121',
    'Zeff_wavelet-LLH_gldm_LargeDependenceLowGrayLevelEmphasis',
    'VMI_DL_139',
    'Zeff_DL_137',
    'IC_wavelet-LHL_firstorder_Skewness',
    'Zeff_wavelet-LHH_glcm_Idn',
    'VMI_wavelet-LLH_glrlm_GrayLevelNonUniformity',
    'VMI_wavelet-HHH_glcm_Correlation',
    'PEI_wavelet-LLH_glszm_SmallAreaEmphasis',
    'IC_wavelet-LLL_glszm_SizeZoneNonUniformity',
    'PEI_DL_243',
    'Histological grade',
    'Clinical T stage'
]

# ===============================
# 3. Default values
# ===============================
default_values = {
    'IC_DL_57': 1.956566,
    'VMI_original_glszm_SmallAreaHighGrayLevelEmphasis': 1.775329,
    'IC_wavelet-LLH_glszm_ZoneEntropy': 3.246436,
    'Zeff_DL_91': -0.0617,
    'VMI_DL_45': 0.626165,
    'IC_DL_121': 2.349622,
    'Zeff_wavelet-LLH_gldm_LargeDependenceLowGrayLevelEmphasis': 1.729691,
    'VMI_DL_139': 0.466549,
    'Zeff_DL_137': 1.65199,
    'IC_wavelet-LHL_firstorder_Skewness': 0.684704,
    'Zeff_wavelet-LHH_glcm_Idn': 3.454122,
    'VMI_wavelet-LLH_glrlm_GrayLevelNonUniformity': 1.927121,
    'VMI_wavelet-HHH_glcm_Correlation': 12.209791,
    'PEI_wavelet-LLH_glszm_SmallAreaEmphasis': 7.412239,
    'IC_wavelet-LLL_glszm_SizeZoneNonUniformity': 4.892986,
    'PEI_DL_243': 1.844251,
    'Histological grade': 1,
    'Clinical T stage': 1
}

# ===============================
# 4. Streamlit page
# ===============================
st.title('Web Predictor for Occult LNM in Patients with HNSCC')
st.sidebar.header('Input Features')

# ===============================
# 5. Input features
# ===============================
inputs = {}

continuous_features = [
    'IC_DL_57',
    'VMI_original_glszm_SmallAreaHighGrayLevelEmphasis',
    'IC_wavelet-LLH_glszm_ZoneEntropy',
    'Zeff_DL_91',
    'VMI_DL_45',
    'IC_DL_121',
    'Zeff_wavelet-LLH_gldm_LargeDependenceLowGrayLevelEmphasis',
    'VMI_DL_139',
    'Zeff_DL_137',
    'IC_wavelet-LHL_firstorder_Skewness',
    'Zeff_wavelet-LHH_glcm_Idn',
    'VMI_wavelet-LLH_glrlm_GrayLevelNonUniformity',
    'VMI_wavelet-HHH_glcm_Correlation',
    'PEI_wavelet-LLH_glszm_SmallAreaEmphasis',
    'IC_wavelet-LLL_glszm_SizeZoneNonUniformity',
    'PEI_DL_243'
]

# Continuous variables
for feature in continuous_features:
    inputs[feature] = st.sidebar.number_input(
        label=feature,
        min_value=-100.0,
        max_value=100.0,
        value=float(default_values[feature]),
        step=0.01,
        format="%.6f"
    )

# ===============================
# Histological grade
# 0: Well
# 1: Moderate
# 2: Poor
# ===============================
histological_options = ['Well', 'Moderate', 'Poor']

histological_grade = st.sidebar.selectbox(
    'Histological grade',
    options=histological_options,
    index=int(default_values['Histological grade'])
)

histological_map = {
    'Well': 0,
    'Moderate': 1,
    'Poor': 2
}

inputs['Histological grade'] = histological_map[histological_grade]

# ===============================
# Clinical T stage
# 0: T1-2
# 1: T3-4
# ===============================
clinical_t_options = ['T1-2', 'T3-4']

clinical_t_stage = st.sidebar.selectbox(
    'Clinical T stage',
    options=clinical_t_options,
    index=int(default_values['Clinical T stage'])
)

clinical_t_map = {
    'T1-2': 0,
    'T3-4': 1
}

inputs['Clinical T stage'] = clinical_t_map[clinical_t_stage]

# Convert input values into DataFrame
input_df = pd.DataFrame([inputs])

# Ensure the order is exactly the same as training
input_df = input_df[feature_label]

# ===============================
# Optional: show input data
# ===============================
# st.write(input_df)

# ===============================
# 6. Prediction
# ===============================
if st.sidebar.button('Predict'):
    try:
        # -------------------------------------------------
        # Important:
        # The model was trained with feature names:
        # ['a', 'b', ..., 'r']
        #
        # Here we use input_df.values to remove feature names,
        # and validate_features=False to bypass name checking.
        #
        # This requires that the feature order is exactly the same
        # as in the training data.
        # -------------------------------------------------

        input_data = xgb.DMatrix(input_df.values)

        prediction = model_xgb.predict(
            input_data,
            validate_features=False
        )[0]

        st.subheader('Predicted probability of occult LNM')

        st.markdown(
            f"""
            <p style="font-size:24px; font-weight:bold;">
                Predicted probability:
                <span style="color:red;">{prediction:.6f}</span>
            </p>
            """,
            unsafe_allow_html=True
        )

        # ===============================
        # 7. SHAP explanation
        # ===============================
        st.subheader('SHAP Force Plot')

        explainer = shap.TreeExplainer(model_xgb)

        # Use values only to avoid feature name mismatch in SHAP
        shap_values = explainer.shap_values(input_df.values)

        plt.figure()

        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_df.iloc[0, :],
            feature_names=feature_label,
            matplotlib=True,
            contribution_threshold=0.1,
            show=False
        )

        plt.savefig(
            "shap_force_plot.png",
            bbox_inches='tight',
            dpi=300
        )

        plt.close()

        st.image("shap_force_plot.png")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
