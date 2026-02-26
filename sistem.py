# ==========================================
# REGRESI LINEAR BERGANDA - PREDIKSI BIAYA ASURANSI
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD DATASET
# ==========================================
df = pd.read_csv("archive/insurance.csv")

print("===== DATA AWAL =====")
print(df.head())
print("\nINFO DATASET")
print(df.info())


# ==========================================
# 2. VISUALISASI DESKRIPSI DATA
# ==========================================

# Distribusi biaya asuransi
plt.figure()
df['charges'].hist(bins=30)
plt.title('Distribusi Biaya Asuransi')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()

# Distribusi umur
plt.figure()
df['age'].hist(bins=30)
plt.title('Distribusi Umur')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Korelasi antar fitur
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm') # Tambahkan numeric_only
plt.title('Korelasi Antar Fitur')
plt.show()


# ==========================================
# 3. CEK MISSING VALUE
# ==========================================
print("\n===== CEK MISSING VALUE =====")
print(df.isnull().sum())


# ==========================================
# 4. ENCODING DATA KATEGORI
# ==========================================
df = pd.get_dummies(df, drop_first=True)

print("\n===== DATA SETELAH ENCODING =====")
print(df.head())


# ==========================================
# 5. SPLIT DATA
# ==========================================
X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 6. SCALING TIDAK DIPERLUKAN UNTUK RANDOM FOREST
# ==========================================

# ==========================================
# 7. TRAINING MODEL (RANDOM FOREST)
# ==========================================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# 8. PREDIKSI
# ==========================================
y_pred = model.predict(X_test)


# ==========================================
# 9. EVALUASI MODEL
# ==========================================
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm') # Tambahkan numeric_only
plt.title('Korelasi Antar Fitur')
plt.show()


# ==========================================
# 10. KOEFISIEN REGRESI
# ==========================================
coef = pd.DataFrame({
    "Fitur": df.drop("charges", axis=1).columns,
    "Koefisien": model.coef_
})

coef = coef.sort_values(by="Koefisien", ascending=False)

print("\n===== KOEFISIEN REGRESI =====")
print(coef)


# ==========================================
# 11. VISUALISASI PREDIKSI VS ASLI
# ==========================================
plt.figure()
plt.scatter(y_test, y_pred)
plt.title('Prediksi vs Aktual')
plt.xlabel('Aktual')
plt.ylabel('Prediksi')
plt.show()


# ==========================================
# SELESAI
# ==========================================