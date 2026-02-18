# ==============================
# REGRESI LINEAR BERGANDA
# Prediksi Biaya Asuransi
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("archive/insurance.csv")

print("Data Awal:")
print(df.head())

# ==============================
# 2. Cek Missing Value
# ==============================

print("\nCek Missing Value:")
print(df.isnull().sum())

# ==============================
# 3. Encoding Data Kategori
# ==============================

df = pd.get_dummies(df, drop_first=True)

print("\nData Setelah Encoding:")
print(df.head())

# ==============================
# 4. Split Data
# ==============================

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 5. Training Model
# ==============================

model = LinearRegression()
model.fit(X_train, y_train)

# ==============================
# 6. Prediksi
# ==============================

y_pred = model.predict(X_test)

# ==============================
# 7. Evaluasi Model
# ==============================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== HASIL EVALUASI =====")
print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2)

# ==============================
# 8. Koefisien Regresi
# ==============================

coef = pd.DataFrame({
    "Fitur": X.columns,
    "Koefisien": model.coef_
})

print("\n===== KOEFISIEN REGRESI =====")
print(coef)

# ==============================
# 9. Grafik Prediksi vs Aktual
# ==============================

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Nilai Aktual")
plt.ylabel("Nilai Prediksi")
plt.title("Grafik Prediksi vs Aktual")
plt.show()

# ==============================
# 10. Grafik Residual
# ==============================

residual = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residual)
plt.xlabel("Nilai Prediksi")
plt.ylabel("Residual")
plt.title("Grafik Residual")
plt.axhline(y=0)
plt.show()