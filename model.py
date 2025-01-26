import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Select Language
language = st.selectbox("Pilih Bahasa / Select Language", ['Bahasa Indonesia', 'English'])

# Set title and messages based on selected language
if language == 'Bahasa Indonesia':
    title = '💼 Kalkulator Estimasi Biaya Kesehatan Menggunakan Algoritma Polynominal Regression'
    success_message = '💵 Perkiraan biaya kesehatan tahunan Anda: **{estimated_cost}**'  # Bahasa Indonesia message
else:
    title = '💼 Health Cost Estimation Calculator Using the Regression Polynomial Algorithm'
    success_message = '💵 Your estimated annual health cost: **{estimated_cost}**'  # English message

# Load data
data = pd.read_csv('Regression.csv')
data = pd.get_dummies(data, drop_first=True)
X = data.drop(columns=['charges'])
y = data['charges']

# Create model
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
poly_model.fit(X_train, y_train)

def predict_charges(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    return poly_model.predict(input_df)[0]

# Texts in Bahasa Indonesia or English based on selection
if language == 'Bahasa Indonesia':
    age_label = '🧓 Usia'
    bmi_label = '⚖️ BMI'
    children_label = '👶 Jumlah Anak'
    smoker_label = '🚬 Apakah Anda Perokok?'
    sex_label = '⚥ Jenis Kelamin'
    region_label = '🌍 Wilayah'
    button_label = '🔍 Hitung Estimasi Biaya'
else:
    age_label = '🧓 Age'
    bmi_label = '⚖️ BMI'
    children_label = '👶 Number of Children'
    smoker_label = '🚬 Are You a Smoker?'
    sex_label = '⚥ Gender'
    region_label = '🌍 Region'
    button_label = '🔍 Calculate Estimated Cost'

# Streamlit UI with dynamic labels based on language selection
st.markdown(f"<h1 style='text-align:left;'>{title}</h1>", unsafe_allow_html=True)  # Title aligned left
st.markdown("<hr style='border-top: 2px solid #f0f0f0;'>", unsafe_allow_html=True)

# Background image and styling
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #6e7dff, #4caf50); /* Beautiful gradient background */
            color: #ffffff;
            font-family: 'Arial', sans-serif;
        } 
        .css-1d391kg {
            background-color: #f7f7f7;
        }
        .css-12oz5g7 {
            font-size: 16px;
        }
        .css-1q8dd3e-Input {
            background-color: #f1f1f1;
            padding: 12px;
        }
        .css-1gw2v94 {
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        h1 {
            font-size: 36px;
            color: #fff;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            padding: 10px;
            font-size: 18px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Using columns for layout
col1, col2 = st.columns(2)
with col1:
    age = st.number_input(age_label, min_value=18, max_value=100, value=30)
    bmi = st.number_input(bmi_label, min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input(children_label, min_value=0, max_value=10, value=0)
with col2:
    smoker = st.selectbox(smoker_label, ['no', 'yes'])
    sex = st.selectbox(sex_label, ['male', 'female'])
    region = st.selectbox(region_label, ['southeast', 'southwest', 'northwest', 'northeast'])

# Adding currency selection above the button
currency = st.selectbox("Pilih Mata Uang / Select Currency", ['USD', 'IDR'])

# Multiple predictions feature
multiple_predictions = st.checkbox('🔢 Multiple Predictions' if language == 'English' else '🔢 Prediksi untuk Beberapa Kasus')

if multiple_predictions:
    st.write('Enter data for multiple users.' if language == 'English' else 'Masukkan data untuk beberapa pengguna.')
    ages = st.text_area("Age (Separate with commas)" if language == 'English' else "Usia (Pisahkan dengan koma)", "30, 40, 50")
    bmis = st.text_area("BMI (Separate with commas)" if language == 'English' else "BMI (Pisahkan dengan koma)", "25, 28, 30")
    children_count = st.text_area("Number of Children (Separate with commas)" if language == 'English' else "Jumlah Anak (Pisahkan dengan koma)", "1, 2, 3")
    
    ages = list(map(int, ages.split(',')))
    bmis = list(map(float, bmis.split(',')))
    children_count = list(map(int, children_count.split(',')))

    if st.button('Calculate All Predictions' if language == 'English' else 'Hitung Semua Prediksi'):
        for i in range(len(ages)):
            user_input = {
                'age': ages[i],
                'bmi': bmis[i],
                'children': children_count[i],
                'smoker_yes': 1 if smoker == 'yes' else 0,
                'sex_male': 1 if sex == 'male' else 0,
                'region_northwest': 1 if region == 'northwest' else 0,
                'region_southeast': 1 if region == 'southeast' else 0,
                'region_southwest': 1 if region == 'southwest' else 0,
            }
            estimated_cost = predict_charges(user_input)
            if currency == 'IDR':
                estimated_cost_in_rupiah = estimated_cost * 15000  # Example conversion rate
                estimated_cost_label = f"Rp {estimated_cost_in_rupiah:,.2f}"
            else:
                estimated_cost_label = f"${estimated_cost:,.2f}"
            st.write(f"**Prediction for Age {ages[i]}**: {estimated_cost_label}")
    
else:
    if st.button(button_label):
        user_input = {
            'age': age,
            'bmi': bmi,
            'children': children,
            'smoker_yes': 1 if smoker == 'yes' else 0,
            'sex_male': 1 if sex == 'male' else 0,
            'region_northwest': 1 if region == 'northwest' else 0,
            'region_southeast': 1 if region == 'southeast' else 0,
            'region_southwest': 1 if region == 'southwest' else 0,
        }
        estimated_cost = predict_charges(user_input)
        
        if currency == 'IDR':
            estimated_cost_in_rupiah = estimated_cost * 15000  # Example conversion rate
            estimated_cost_label = f"Rp {estimated_cost_in_rupiah:,.2f}"
        else:
            estimated_cost_label = f"${estimated_cost:,.2f}"
        
        st.success(success_message.format(estimated_cost=estimated_cost_label))

        # Recommendation and region comparison
        if estimated_cost > 10000:
            st.warning('💡 Based on the high estimated cost, it is recommended to consider health insurance with larger coverage.' if language == 'English' else '💡 Berdasarkan estimasi biaya yang tinggi, disarankan untuk mempertimbangkan asuransi kesehatan dengan cakupan lebih besar.')
        elif estimated_cost < 5000:
            st.info('💡 Your cost is relatively low, but make sure you have insurance to protect against unforeseen risks.' if language == 'English' else '💡 Biaya Anda relatif rendah, namun pastikan Anda memiliki asuransi untuk melindungi dari risiko tak terduga.')

        # Cost Comparison Based on Region
        if language == 'English':
            region_comparison = {
                'northwest': 1.1,  # Cost factor by region
                'southwest': 0.9,
                'southeast': 1.05,
                'northeast': 1.0
            }
            region_cost_factor = region_comparison.get(region, 1.0)
            adjusted_cost = estimated_cost * region_cost_factor
            st.write(f"💡 Your estimated cost after considering region {region}: {estimated_cost_label}")
        else:
            region_comparison = {
                'northwest': 1.1,  # Faktor biaya berdasarkan wilayah
                'southwest': 0.9,
                'southeast': 1.05,
                'northeast': 1.0
            }
            region_cost_factor = region_comparison.get(region, 1.0)
            adjusted_cost = estimated_cost * region_cost_factor
            st.write(f"💡 Perkiraan biaya Anda setelah mempertimbangkan wilayah {region}: {estimated_cost_label}")

# Footer with styling
st.markdown("<hr style='border-top: 2px solid #f0f0f0;'>", unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color: #6c757d;">© 2024 Health Cost Estimator | Ria Maisya Syarifah</p>', unsafe_allow_html=True)
