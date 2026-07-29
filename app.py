# -*- coding: utf-8 -*-
"""
California Housing Price Predictor
Streamlit Web Application
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="🏠 California Housing Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    /* พื้นหลังหลัก - สีอ่อน */
    .main {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
    }
    
    /* ตัวอักษรทั้งหมด - สีดำ */
    * {
        color: #2c3e50 !important;
    }
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
        font-weight: 700 !important;
    }
    
    /* Slider labels */
    .stSlider label {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Slider values */
    .stSlider [data-testid="stMarkdownContainer"] p {
        color: #e74c3c !important;
        font-weight: bold !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card h3 {
        color: #667eea !important;
        margin: 0 !important;
    }
    
    .metric-card p {
        color: #2c3e50 !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white !important;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .result-card * {
        color: white !important;
    }
    
    .result-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* Dataframe */
    .dataframe {
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Load Model ====================
@st.cache_resource
def load_model():
    """โหลดโมเดลและ scaler จากไฟล์"""
    try:
        model = joblib.load('model_files/rf_model.pkl')
        scaler = joblib.load('model_files/scaler.pkl')
        feature_names = joblib.load('model_files/feature_names.pkl')
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
        st.stop()

model, scaler, feature_names = load_model()

# ==================== Sidebar ====================
with st.sidebar:
    st.markdown("## 🏠 California Housing")
    st.markdown("### Price Predictor")
    st.markdown("---")
    st.markdown("""
    **โมเดล:** Random Forest Regressor  
    **Dataset:** California Housing  
    **Accuracy:** R² = 0.81
    
    ---
    พัฒนาโดย: นักศึกษาวิชา ML with Python
    """)
    
    # Feature descriptions
    with st.expander(" คำอธิบาย Features"):
        st.markdown("""
        - **MedInc**: รายได้เฉลี่ย (×$10,000)
        - **HouseAge**: อายุบ้านเฉลี่ย
        - **AveRooms**: จำนวนห้องเฉลี่ย
        - **AveBedrms**: จำนวนห้องนอนเฉลี่ย
        - **Population**: จำนวนประชากร
        - **AveOccup**: จำนวนคนต่อครัวเรือน
        - **Latitude**: ละติจูด
        - **Longitude**: ลองจิจูด
        """)

# ==================== Main Content ====================
st.markdown("""
<div style='text-align: center; padding: 1rem 0;'>
    <h1 style='color: #2c3e50; font-size: 3rem; margin-bottom: 0.5rem;'>
        🏠 California Housing Price Predictor
    </h1>
    <p style='color: #7f8c8d; font-size: 1.2rem;'>
        ทำนายราคาบ้านในแคลิฟอร์เนียด้วย Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==================== Input Section ====================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📍 Location & Demographics")
    latitude = st.slider("🌐 Latitude", 32.0, 42.0, 35.0, 0.01)
    longitude = st.slider("🌐 Longitude", -124.0, -114.0, -119.0, 0.01)
    population = st.slider("👥 Population", 1.0, 30000.0, 1500.0, 10.0)
    ave_occup = st.slider("👨‍👩👧‍👦 Avg. Occupancy", 1.0, 10.0, 3.0, 0.1)

with col2:
    st.markdown("### 🏡 House Characteristics")
    med_inc = st.slider("💰 Median Income (×$10k)", 0.5, 15.0, 5.0, 0.1)
    house_age = st.slider("🏚️ House Age (years)", 1.0, 52.0, 20.0, 1.0)
    ave_rooms = st.slider(" Avg. Rooms", 1.0, 15.0, 5.0, 0.1)
    ave_bedrms = st.slider("🛏️ Avg. Bedrooms", 0.5, 5.0, 1.1, 0.1)

# ==================== Prediction Button ====================
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_button = st.button("🔮 ทำนายราคาบ้าน", use_container_width=True)

# ==================== Prediction Logic ====================
if predict_button:
    # สร้าง input array
    input_data = np.array([[
        med_inc, house_age, ave_rooms, ave_bedrms,
        population, ave_occup, latitude, longitude
    ]])
    
    # Scale ข้อมูล
    input_scaled = scaler.transform(input_data)
    
    # ทำนาย
    prediction = model.predict(input_scaled)[0]
    prediction_usd = prediction * 100000
    
    # แสดงผลลัพธ์
    st.markdown("---")
    st.markdown(f"""
    <div class='result-card'>
        <h2 style='margin: 0;'>💰 ราคาบ้านที่ทำนายได้</h2>
        <div class='result-value'>${prediction_usd:,.0f}</div>
        <p style='font-size: 1.1rem; margin: 0;'>
            หรือประมาณ {prediction:,.2f} × $100,000
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # แสดงรายละเอียด
    st.markdown("### 📊 รายละเอียดการทำนาย")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #667eea; margin: 0;'> Predicted Price</h3>
            <p style='font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;'>
                ${prediction_usd:,.0f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        price_per_room = prediction_usd / ave_rooms if ave_rooms > 0 else 0
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #764ba2; margin: 0;'>🚪 Price/Room</h3>
            <p style='font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;'>
                ${price_per_room:,.0f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        price_per_person = prediction_usd / population if population > 0 else 0
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #f093fb; margin: 0;'> Price/Person</h3>
            <p style='font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;'>
                ${price_per_person:,.0f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        confidence = 81  # R² score
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #4facfe; margin: 0;'>🎯 Confidence</h3>
            <p style='font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;'>
                {confidence}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Importance Chart
    st.markdown("### 📈 Feature Importance")
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis',
            title='🔍 ความสำคัญของ Features ในการทำนาย'
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Input Summary
    with st.expander("📋 ดูข้อมูล Input ที่ใช้ทำนาย"):
        input_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': input_data[0]
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)

# ==================== Footer ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    <p>🎓 พัฒนาเป็นส่วนหนึ่งของวิชา การโปรแกรมสำหรับการเรียนรู้ด้วยเครื่องด้วยภาษาไพทอน</p>
    <p>📅 กรกฎาคม 2026 | 🛠️ Built with Streamlit + scikit-learn</p>
</div>
""", unsafe_allow_html=True)