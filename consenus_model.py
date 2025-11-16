import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# --- Bagian 1: Inisialisasi dan Pelatihan Model Konsensus ---

# 1. Membuat Data Dummy Sederhana (Contoh: Memprediksi Apakah Seseorang Suka Kopi)
# Fitur: Umur, Jam Tidur, Level Kafein Harian (semua disederhanakan)
data = {
    'Umur': [25, 30, 45, 22, 55, 33, 40, 28, 60, 19],
    'Jam_Tidur': [7, 6, 8, 5, 7, 6, 7, 8, 5, 9],
    'Level_Kafein': [3, 5, 1, 6, 0, 4, 2, 5, 0, 4],
    'Suka_Kopi': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1] # 1 = Suka, 0 = Tidak Suka
}
df = pd.DataFrame(data)

# Persiapan Data
X = df[['Umur', 'Jam_Tidur', 'Level_Kafein']]
y = df['Suka_Kopi']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Mendefinisikan Tiga Model AI Individual (Tugas AI yang Berbeda)
model_logreg = LogisticRegression(random_state=42) # Model 1: Linear
model_dtree = DecisionTreeClassifier(random_state=42) # Model 2: Berbasis Pohon Keputusan
model_logreg_slow = LogisticRegression(C=0.1, random_state=42) # Model 3: Varian Model Linear

# 3. Membuat Model Konsensus (Voting Classifier)
# 'soft' voting menggunakan probabilitas untuk konsensus yang lebih halus
consensus_model = VotingClassifier(
    estimators=[
        ('lr', model_logreg),
        ('dt', model_dtree),
        ('lr_slow', model_logreg_slow)
    ],
    voting='soft'
)

# 4. Melatih Semua Model (Pelatihan Model Konsensus)
consensus_model.fit(X_train, y_train)

# --- Bagian 2: Tampilan Web Streamlit ---

st.set_page_config(page_title="🤖 AI Konsensus Sederhana", layout="wide")
st.title("🤝 AI Konsensus: Penentu Pemungutan Suara")

st.markdown("""
Aplikasi ini menunjukkan bagaimana tiga model AI yang berbeda bekerja sama
untuk menentukan satu jawaban (memprediksi apakah seseorang suka kopi atau tidak).
""")

# 5. Input Pengguna
st.subheader("Masukkan Profil Anda")
col1, col2, col3 = st.columns(3)

with col1:
    user_age = st.slider("Umur (Tahun)", 18, 70, 30)
with col2:
    user_sleep = st.slider("Jam Tidur Rata-Rata (Jam)", 4, 10, 7)
with col3:
    user_caffeine = st.slider("Level Kafein Harian (Skala 0-7)", 0, 7, 3)

# Membuat DataFrame untuk prediksi
user_data = pd.DataFrame({
    'Umur': [user_age],
    'Jam_Tidur': [user_sleep],
    'Level_Kafein': [user_caffeine]
})

if st.button("Tentukan Konsensus"):
    # 6. Prediksi oleh Setiap Model Individual
    individual_preds = {}
    
    # Model 1
    pred_lr = model_logreg.predict_proba(user_data)[:, 1][0]
    individual_preds['Model_Linear'] = round(pred_lr, 2)
    
    # Model 2
    pred_dt = model_dtree.predict_proba(user_data)[:, 1][0]
    individual_preds['Model_Pohon'] = round(pred_dt, 2)
    
    # Model 3
    pred_lrslow = model_logreg_slow.predict_proba(user_data)[:, 1][0]
    individual_preds['Model_Linear_Varian'] = round(pred_lrslow, 2)

    # 7. Prediksi Konsensus (Final)
    consensus_proba = consensus_model.predict_proba(user_data)[:, 1][0]
    final_prediction = "Suka Kopi" if consensus_proba >= 0.5 else "Tidak Suka Kopi"
    
    # 8. Tampilkan Hasil
    st.markdown("---")
    st.subheader("Hasil Prediksi Konsensus")
    
    # Hasil Final
    if consensus_proba >= 0.5:
        st.balloons()
        st.success(f"**Keputusan Konsensus:** {final_prediction} (Probabilitas: {consensus_proba:.2f})")
    else:
        st.error(f"**Keputusan Konsensus:** {final_prediction} (Probabilitas: {consensus_proba:.2f})")

    st.markdown("---")
    
    # Kontribusi Individual
    st.subheader("Kontribusi Model Individual (Probabilitas 'Suka Kopi')")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Model 1 (Linear)", f"{individual_preds['Model_Linear']:.2f}")
    col_b.metric("Model 2 (Pohon Keputusan)", f"{individual_preds['Model_Pohon']:.2f}")
    col_c.metric("Model 3 (Linear Varian)", f"{individual_preds['Model_Linear_Varian']:.2f}")

    st.info(f"Keputusan Konsensus didasarkan pada **rata-rata probabilitas tertimbang** dari ketiga model.")
