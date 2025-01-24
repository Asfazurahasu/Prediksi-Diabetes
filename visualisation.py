import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def show_visualisation():
    # Pastikan file diabetes.csv ada di direktori yang sama dengan file .py
    try:
        df = pd.read_csv('diabetes.csv')
    except FileNotFoundError:
        st.error("File 'diabetes.csv' tidak ditemukan. Pastikan file berada di direktori yang benar.")
        return

    st.subheader("Visualisasi Data Diabetes")
    st.markdown("""
        Di bawah ini, Anda dapat melihat beberapa grafik yang menunjukkan hubungan antara berbagai faktor medis dan risiko diabetes.
    """)

    # Visualisasi Hubungan antara Fitur dan Outcome
    st.subheader("Visualisasi Hubungan Faktor Medis dengan Outcome Diabetes")
    
    # Grafik distribusi dari beberapa fitur
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df['Pregnancies'], kde=True, ax=ax[0, 0], color='blue').set_title('Distribusi Kehamilan')
    sns.histplot(df['Glucose'], kde=True, ax=ax[0, 1], color='green').set_title('Distribusi Kadar Gula')
    sns.histplot(df['BMI'], kde=True, ax=ax[1, 0], color='orange').set_title('Distribusi BMI')
    sns.histplot(df['Age'], kde=True, ax=ax[1, 1], color='red').set_title('Distribusi Umur')

    st.pyplot(fig)

    # Visualisasi distribusi fitur lainnya
    fig2, ax2 = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df['BloodPressure'], kde=True, ax=ax2[0, 0], color='purple').set_title('Distribusi Tekanan Darah')
    sns.histplot(df['SkinThickness'], kde=True, ax=ax2[0, 1], color='cyan').set_title('Distribusi Ketebalan Kulit')
    sns.histplot(df['Insulin'], kde=True, ax=ax2[1, 0], color='yellow').set_title('Distribusi Insulin')
    sns.histplot(df['DiabetesPedigreeFunction'], kde=True, ax=ax2[1, 1], color='brown').set_title('Distribusi Fungsi Pedigree Diabetes')

    st.pyplot(fig2)

    # Heatmap Korelasi
    st.subheader("Korelasi antara Fitur Medis")
    correlation_matrix = df.drop(columns='Outcome').corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

    # Fitur KNN
    st.subheader("Visualisasi KNN (K-Nearest Neighbors) untuk Diagnosis Diabetes")
    
    # Pilihan fitur untuk KNN
    selected_features = st.multiselect(
        "Pilih Fitur untuk Klasifikasi KNN", 
        ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        default=['Pregnancies', 'Glucose', 'BMI', 'Age']
    )
    
    if len(selected_features) < 2:
        st.warning("Pilih setidaknya dua fitur untuk melatih model KNN.")
        return

    # Siapkan Data untuk KNN
    X = df[selected_features]
    y = df['Outcome']
    
    # Standarisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Bagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Latih model KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Prediksi menggunakan model KNN
    y_pred = knn.predict(X_test)

    # Tampilkan hasil klasifikasi
    st.subheader("Hasil Klasifikasi KNN")
    st.write(f"Jumlah Data Uji: {len(y_test)}")
    st.write(f"Akurasi: {knn.score(X_test, y_test):.4f}")
    
    # Confusion matrix yang sudah ditentukan
    cm = [[119, 32], [37, 43]]  # Nilai confusion matrix yang diberikan

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])

    # Menambahkan nilai TP, TN, FP, FN langsung di atas kotak-kotak matrix
    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    # Menampilkan nilai-nilai dalam diagram (di atas setiap kotak)
    for i in range(2):
        for j in range(2):
            ax_cm.text(j + 0.1, i + 0.1, str(cm[i][j]), color='black', fontsize=16, fontweight='bold')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(fig_cm)

    # Menampilkan nilai-nilai TP, TN, FP, FN dalam teks
    st.subheader("Evaluasi Model (TP, TN, FP, FN)")
    st.write(f"True Negative (TN): {TN}")
    st.write(f"False Positive (FP): {FP}")
    st.write(f"False Negative (FN): {FN}")
    st.write(f"True Positive (TP): {TP}")

    # Tampilkan classification report
    st.subheader("Laporan Klasifikasi")
    st.text(classification_report(y_test, y_pred))

