import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import pickle
import os

# Nama file model untuk disimpan/dimuat
MODEL_FILE = 'consensus_model.pkl'

# --- Fungsi untuk Melatih dan Menyimpan Model Konsensus ---
def train_and_save_model():
    # Data Dummy untuk Pelatihan (Contoh: Suka Kopi)
    data = {
        'Umur': [25, 30, 45, 22, 55, 33, 40, 28, 60, 19, 35, 29, 50, 27, 42],
        'Jam_Tidur': [7, 6, 8, 5, 7, 6, 7, 8, 5, 9, 6, 7, 7, 6, 8],
        'Level_Kafein': [3, 5, 1, 6, 0, 4, 2, 5, 0, 4, 3, 5, 1, 4, 2],
        'Suka_Kopi': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0] 
    }
    df = pd.DataFrame(data)
    
    X = df[['Umur', 'Jam_Tidur', 'Level_Kafein']]
    y = df['Suka_Kopi']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Mendefinisikan Tiga Model Individual (Expert)
    model_logreg = LogisticRegression(random_state=42) # Model 1: Linear
    model_dtree = DecisionTreeClassifier(random_state=42) # Model 2: Pohon Keputusan
    model_logreg_slow = LogisticRegression(C=0.1, random_state=42) # Model 3: Varian Model Linear
    
    # Membuat Model Konsensus (Voting Classifier)
    consensus_model = VotingClassifier(
        estimators=[
            ('lr', model_logreg),
            ('dt', model_dtree),
            ('lr_slow', model_logreg_slow)
        ],
        voting='soft'
    )
    
    # Melatih Model
    consensus_model.fit(X_train, y_train)
    
    # Menyimpan model ke disk
    with open(MODEL_FILE, 'wb') as file:
        pickle.dump(consensus_model, file)
        
    return consensus_model

# --- Fungsi untuk Memuat Model ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return train_and_save_model()
    
    with open(MODEL_FILE, 'rb') as file:
        model = pickle.load(file)
    return model

# --- Bagian Utama Aplikasi Streamlit ---

# 1. Muat atau Latih Model
model_consensus = load_model()
# Ekstrak model individual untuk menampilkan prediksi detail
individual_models = {name: est for name, est in model_consensus.estimators}

st.set_page_config(page_title="🤖 AI Konsensus Sederhana", layout="wide")
st.title("🤝 AI Konsensus: Penentu Pemungutan Suara")

st.markdown("""
Aplikasi ini menunjukkan bagaimana tiga model AI individu bekerja sama 
menggunakan metode **Voting Classifier** untuk mencapai satu keputusan terbaik.
""")

# 2. Input Pengguna
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

# 3. Tombol Prediksi
if st.button("Tentukan Konsensus", type="primary"):
    
    # 4. Prediksi oleh Setiap Model Individual
    individual_preds = {}
    for name, model in individual_models.items():
        # Memprediksi probabilitas kelas 1 ('Suka Kopi')
        prob = model.predict_proba(user_data)[:, 1][0]
        individual_preds[name] = prob

    # 5. Prediksi Konsensus (Final)
    consensus_proba = model_consensus.predict_proba(user_data)[:, 1][0]
    final_prediction = "Suka Kopi" if consensus_proba >= 0.5 else "Tidak Suka Kopi"
    
    # 6. Tampilkan Hasil
    st.markdown("---")
    st.subheader("🎉 Hasil Keputusan Konsensus")
    
    col_res, col_prob = st.columns(2)
    
    with col_res:
        if final_prediction == "Suka Kopi":
            st.success(f"**Keputusan Final:** {final_prediction}")
            st.balloons()
        else:
            st.error(f"**Keputusan Final:** {final_prediction}")
            
    with col_prob:
        st.metric("Probabilitas Konsensus", f"{consensus_proba*100:.1f}%")

    st.markdown("---")
    
    # Kontribusi Individual
    st.subheader("Kontribusi Model Individual (Probabilitas 'Suka Kopi')")
    
    cols_ind = st.columns(3)
    
    cols_ind[0].metric("Model 1 (Linear)", f"{individual_preds['lr']:.2f}", 
                       help="Menggunakan Regresi Logistik dasar.")
    cols_ind[1].metric("Model 2 (Pohon Keputusan)", f"{individual_preds['dt']:.2f}",
                       help="Menggunakan model yang fokus pada pemisahan data berbasis aturan.")
    cols_ind[2].metric("Model 3 (Varian Linear)", f"{individual_preds['lr_slow']:.2f}",
                       help="Menggunakan Regresi Logistik dengan parameter regularisasi berbeda.")

    st.info("Keputusan Konsensus didasarkan pada rata-rata probabilitas tertimbang (soft voting) dari ketiga model.")
