import streamlit as st

def show_home():
    st.subheader("Mari Mulai Menggunakan Aplikasi Ini!")
    st.markdown("""
        Pilih tab **Data Visualisation** untuk melihat grafik interaktif yang menunjukkan hubungan antara berbagai fitur medis dan risiko diabetes.
        Atau pilih tab **Prediction** untuk mulai memprediksi risiko diabetes Anda berdasarkan data yang Anda masukkan.
    """)
