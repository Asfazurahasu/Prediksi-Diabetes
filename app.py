import streamlit as st
from streamlit_option_menu import option_menu
import home
import visualisation
import prediction

# Set up page configuration
st.set_page_config(page_title="Prediksi Diabetes", page_icon="🍏", layout="wide")

# Add a title and description
st.title("Selamat Datang di Aplikasi Prediksi Diabetes")
st.markdown("Web ini dibuat oleh Asfazura Hasugian (220180042) untuk memenuhi ujian akhir semester machine learning")
st.markdown("""
    Aplikasi ini dirancang untuk membantu Anda mengetahui kemungkinan Anda menderita diabetes berdasarkan beberapa fitur medis.

    **Fitur Utama:**
    - Prediksi risiko diabetes berdasarkan data medis yang Anda masukkan
    - Visualisasi data interaktif untuk analisis faktor-faktor risiko diabetes
    - Dapat digunakan oleh siapa saja yang ingin mengetahui risiko diabetes lebih awal.

    **Cara Penggunaan:**
    Masukkan informasi medis yang relevan, seperti usia, kadar gula darah, BMI, dan lainnya. Setelah itu, aplikasi ini akan memberikan prediksi apakah Anda berisiko terkena diabetes.
""")

# Menambahkan gambar atau logo pada halaman utama
st.image("https://static.vecteezy.com/system/resources/previews/011/132/385/original/world-diabetes-day-logo-design-free-vector.jpg", width=300)

st.markdown("""
    Diabetes adalah salah satu penyakit yang paling umum di dunia dan dapat mempengaruhi kualitas hidup. Namun, dengan gaya hidup sehat dan pemeriksaan rutin, risiko diabetes dapat dikendalikan.

    Aplikasi ini menggunakan teknologi machine learning untuk memberikan prediksi risiko berdasarkan data medis yang Anda masukkan. Mari jelajahi lebih lanjut melalui berbagai visualisasi yang bisa membantu Anda memahami lebih dalam tentang diabetes!
""")

# Membuat menu navigasi antar halaman
selected = option_menu(
    menu_title="Navigasi",  # Judul menu
    options=["Home", "Data Visualisation", "Prediction"],  # Daftar halaman
    icons=["house", "bar-chart", "calculator"],  # Ikon untuk masing-masing halaman
    menu_icon="cast",  # Ikon menu di kiri atas
    default_index=0,  # Halaman default adalah 'Home'
    orientation="horizontal",  # Layout menu horizontal
)

# Menampilkan halaman berdasarkan menu yang dipilih
if selected == "Home":
    home.show_home()
elif selected == "Data Visualisation":
    visualisation.show_visualisation()
elif selected == "Prediction":
    prediction.show_prediction()
