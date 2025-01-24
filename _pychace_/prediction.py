import streamlit as st
import numpy as np
import joblib

# Pastikan model dan scaler sudah disimpan dengan benar
try:
    knn = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model KNN atau Scaler tidak ditemukan. Pastikan file 'knn_model.pkl' dan 'scaler.pkl' ada di direktori yang benar.")
    knn = None
    scaler = None

def show_prediction():
    st.subheader("Prediksi Risiko Diabetes Anda")
    
    # Input data untuk prediksi dari pengguna
    pregnancies = st.number_input('Jumlah Kehamilan (Pregnancies)', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Kadar Gula (Glucose)', min_value=0, max_value=200, value=0)
    blood_pressure = st.number_input('Tekanan Darah (Blood Pressure)', min_value=0, max_value=200, value=0)
    skin_thickness = st.number_input('Ketebalan Kulit (Skin Thickness)', min_value=0, max_value=100, value=0)
    insulin = st.number_input('Kadar Insulin (Insulin)', min_value=0, max_value=1000, value=0)
    bmi = st.number_input('Indeks Massa Tubuh (BMI)', min_value=0.0, max_value=100.0, value=0.0)
    diabetes_pedigree_function = st.number_input('Fungsi Pedigree Diabetes (Diabetes Pedigree Function)', min_value=0.0, max_value=2.5, value=0.0)
    age = st.number_input('Usia (Age)', min_value=0, max_value=100, value=0)

    # Membuat array data untuk prediksi
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    if knn is not None and scaler is not None:
        # Lakukan standarisasi data input berdasarkan scaler yang telah dilatih
        input_data_scaled = scaler.transform(input_data)

        # Prediksi dengan model KNN
        button = st.button('Prediksi Risiko Diabetes')

        if button:
            pred = knn.predict(input_data_scaled)
            if pred == 1:
                st.write("Pasien terkena diabetes.")
            else:
                st.write("Pasien tidak terkena diabetes.")
