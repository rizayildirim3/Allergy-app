import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="Pediatric Allergy Predictor",
    page_icon="👶",
    layout="centered"
)


# 2. Load the Optimized Model
@st.cache_resource
def load_model():
    # Modelin eğitildiği sütun sırasını burada tanımlıyoruz (Kritik nokta)
    return joblib.load('allergy_model_v2_7features.pkl')


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. Header
st.title("👶 Pediatric Food Allergy Risk Analysis")
st.markdown("Binary Classification System based on optimized **AdaBoost** model.")
st.divider()

# 4. User Input Fields
st.subheader("📋 Patient Clinical Data")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Months)", min_value=0, max_value=36, value=24)
    ige = st.number_input("Total IgE (IU/mL)", min_value=0.0, value=150.0)
    lymphocytes = st.number_input("Lymphocyte Count (10^3/L)", min_value=0.0, value=3.5, format="%.2f")
    eosinophils = st.number_input("Eozinofil Count (10^3/L)", min_value=0.0, value=0.4, format="%.2f")

with col2:
    breastfeeding_label = st.selectbox("Breastfeeding", options=["Yes", "No"])
    # Kategorik veriyi modele uygun sayısal değere çeviriyoruz
    breastfeeding = 1 if breastfeeding_label == "Yes" else 0

    lpr = st.number_input("LPR (Lymphocyte/Platelet Ratio)", min_value=0.0, value=0.0200, format="%.4f")
    nlr = st.number_input("NLR (Neutrophil/Lymphocyte Ratio)", min_value=0.0, value=1.5000, format="%.4f")

# 5. Prediction Logic
if st.button("🔍 Run Risk Analysis", use_container_width=True):
    # HATANIN ÇÖZÜMÜ: Sütun isimlerini ve değerlerini eşleştirerek DataFrame oluşturma
    # Modelin Pipeline içindeki ColumnTransformer bu isimleri ve sırayı bekler.
    input_data = {
        'ige': [float(ige)],
        'age': [float(age)],
        'lymphocytes': [float(lymphocytes)],
        'eosinophils': [float(eosinophils)],
        'breastfeeding': [int(breastfeeding)],
        'lpr': [float(lpr)],
        'nlr': [float(nlr)]
    }

    input_df = pd.DataFrame(input_data)

    # Sütun sırasının eğitimdekiyle (ige, age, lymphocytes, eosinophils, breastfeeding, lpr, nlr)
    # birebir aynı olduğundan emin oluyoruz:
    feature_order = ['ige', 'age', 'lymphocytes', 'eosinophils', 'breastfeeding', 'lpr', 'nlr']
    input_df = input_df[feature_order]

    try:
        # Tahmin ve Olasılık Hesaplama
        prob = model.predict_proba(input_df)[0][1]
        prob_percent = prob * 100

        # 6. Visual Gauge (Updated for 0.40 Threshold)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 40], 'color': "#2ecc71"},  # Low Risk (Green)
                    {'range': [40, 100], 'color': "#e74c3c"}  # High Risk (Red)
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 40  # Set at 0.40 as requested
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # 7. Binary Clinical Interpretation
        st.divider()
        if prob >= 0.40:
            st.error(f"### 🚨 HIGH RISK ({prob_percent:.1f}%)")
            st.markdown(
                "**Recommendation:** Clinical findings suggest a high probability of food allergy. Further diagnostic workup (e.g., Skin Prick Test, Oral Food Challenge) is advised.")
        else:
            st.success(f"### ✅ LOW RISK ({prob_percent:.1f}%)")
            st.markdown(
                "**Recommendation:** Probability of food allergy is low. Routine follow-up is suggested unless clinical symptoms persist.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Lütfen model dosyasının ve girdi verilerinin uyumlu olduğundan emin olun.")

# Sidebar
st.sidebar.markdown("### 🔬 Model Specs")
st.sidebar.write("**Threshold Applied:** 0.40")
st.sidebar.write("**Model AUC:** 0.814")
st.sidebar.caption("This tool is for decision support only.")
