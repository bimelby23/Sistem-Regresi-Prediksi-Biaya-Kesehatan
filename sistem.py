# ==========================================
# REGRESI LINEAR BERGANDA
# PREDIKSI BIAYA ASURANSI
# ==========================================

# ==========================================
# 1. IMPORT LIBRARY
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")


# ==========================================
# 2. LOAD DATASET
# ==========================================
df = pd.read_csv("archive/insurance.csv")

print("===== DATA AWAL =====")
print(df.head())

print("\n===== INFO DATASET =====")
print(df.info())


# ==========================================
# 3. DESKRIPSI STATISTIK DATA
# ==========================================
print("\n===== DESKRIPSI STATISTIK =====")
print(df.describe())


# ==========================================
# 4. VISUALISASI DATA
# ==========================================

# Distribusi biaya asuransi
plt.figure(figsize=(6,4))
sns.histplot(df['charges'], kde=True)
plt.title("Distribusi Biaya Asuransi")
plt.show()

# Distribusi umur
plt.figure(figsize=(6,4))
sns.histplot(df['age'], kde=True)
plt.title("Distribusi Umur")
plt.show()

# Distribusi BMI
plt.figure(figsize=(6,4))
sns.histplot(df['bmi'], kde=True)
plt.title("Distribusi BMI")
plt.show()

# Hubungan BMI dengan Charges
plt.figure(figsize=(6,4))
sns.scatterplot(x="bmi", y="charges", data=df)
plt.title("Hubungan BMI dengan Biaya Asuransi")
plt.show()

# Charges berdasarkan Smoker
plt.figure(figsize=(6,4))
sns.boxplot(x="smoker", y="charges", data=df)
plt.title("Biaya Asuransi Berdasarkan Status Merokok")
plt.show()


# ==========================================
# 5. HEATMAP KORELASI
# ==========================================
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Korelasi Antar Variabel")
plt.show()


# ==========================================
# 6. CEK MISSING VALUE
# ==========================================
print("\n===== CEK MISSING VALUE =====")
print(df.isnull().sum())


# ==========================================
# 7. ENCODING DATA KATEGORI
# ==========================================
df = pd.get_dummies(df, drop_first=True)

print("\n===== DATA SETELAH ENCODING =====")
print(df.head())


# ==========================================
# 8. SPLIT DATA
# ==========================================
X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 9. SCALING DATA
# ==========================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================
# 10. TRAINING MODEL
# REGRESI LINEAR BERGANDA
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)


# ==========================================
# 11. PREDIKSI
# ==========================================
y_pred = model.predict(X_test)


# ==========================================
# 12. EVALUASI MODEL
# ==========================================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== EVALUASI MODEL =====")
print("MAE  :", mae)
print("RMSE :", rmse)
print("R2   :", r2)


# ==========================================
# 13. KOEFISIEN REGRESI
# ==========================================
coef = pd.DataFrame({
    "Fitur": X.columns,
    "Koefisien": model.coef_
})

coef = coef.sort_values(by="Koefisien", ascending=False)

print("\n===== KOEFISIEN REGRESI =====")
print(coef)


# ==========================================
# 14. VISUALISASI PREDIKSI VS AKTUAL
# ==========================================
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red'
)

plt.xlabel("Nilai Aktual")
plt.ylabel("Nilai Prediksi")
plt.title("Perbandingan Nilai Aktual vs Prediksi")
plt.show()


# ==========================================
# SELESAI
# ==========================================
