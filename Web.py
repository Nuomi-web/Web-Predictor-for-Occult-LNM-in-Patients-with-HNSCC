import streamlit as st
import joblib
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# ===============================
# 1. Load XGBoost model
# ===============================
model_xgb = joblib.load('XGBoost.pkl')

# ===============================
# 2. Feature names
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
# 3. Streamlit page setting
# ===============================
st.title('Web Predictor for Occult LNM in Patients with HNSCC')
st.sidebar.header('Input Features')

# ===============================
# 4. Input features
# ===============================
inputs = {}

# Continuous imaging/radiomics/deep learning features
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

for feature in continuous_features:
    inputs[feature] = st.sidebar.number_input(
        label=feature,
        min_value=-100.0,
        max_value=100.0,
        value=0.0,
        step=0.01
    )

# Categorical clinical feature 1:
# Histological grade: 0 Well, 1 Moderate, 2 Poor
histological_grade = st.sidebar.selectbox(
    'Histological grade',
    options=['Well', 'Moderate', 'Poor']
)

if histological_grade == 'Well':
    inputs['Histological grade'] = 0
elif histological_grade == 'Moderate':
    inputs['Histological grade'] = 1
else:
    inputs['Histological grade'] = 2

# Categorical clinical feature 2:
# Clinical T stage: 0 T1-2, 1 T3-4
clinical_t_stage = st.sidebar.selectbox(
    'Clinical T stage',
    options=['T1-2', 'T3-4']
)

if clinical_t_stage == 'T1-2':
    inputs['Clinical T stage'] = 0
else:
    inputs['Clinical T stage'] = 1

# Convert input values into DataFrame
input_df = pd.DataFrame([inputs])

# Ensure the feature order is exactly the same as training
input_df = input_df[feature_label]

# ===============================
# 5. Prediction
# ===============================
if st.sidebar.button('Predict'):
    try:
        # XGBoost native Booster requires DMatrix
        input_data = xgb.DMatrix(input_df)

        prediction = model_xgb.predict(input_data)[0]

        st.subheader('Predicted Possibility of Occult LNM')
        st.write(f'Predicted Value: {prediction:.4f}')

        # ===============================
        # 6. SHAP explanation
        # ===============================
        st.subheader('SHAP Force Plot')

        explainer = shap.TreeExplainer(model_xgb)
        shap_values = explainer.shap_values(input_df)

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
            dpi=1200
        )

        plt.close()

        st.image("shap_force_plot.png")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")