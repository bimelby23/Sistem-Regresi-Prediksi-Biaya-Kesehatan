# ==========================================
# REGRESI LINEAR BERGANDA - PREDIKSI BIAYA ASURANSI
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================
df = pd.read_csv("archive/insurance.csv")

print("===== DATA AWAL =====")
print(df.head())


# ==========================================
# 2. CEK MISSING VALUE
# ==========================================
print("\n===== CEK MISSING VALUE =====")
print(df.isnull().sum())


# ==========================================
# 3. ENCODING DATA KATEGORI
# ==========================================
df = pd.get_dummies(df, drop_first=True)

print("\n===== DATA SETELAH ENCODING =====")
print(df.head())


# ==========================================
# 4. SPLIT DATA
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
# 5. TRAINING MODEL
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)


# ==========================================
# 6. PREDIKSI
# ==========================================
y_pred = model.predict(X_test)


# ==========================================
# 7. EVALUASI MODEL
# ==========================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== HASIL EVALUASI MODEL =====")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")


# ==========================================
# 8. KOEFISIEN REGRESI
# ==========================================
coef = pd.DataFrame({
    "Fitur": X.columns,
    "Koefisien": model.coef_
})

coef = coef.sort_values(by="Koefisien", ascending=False)

print("\n===== KOEFISIEN REGRESI =====")
print(coef)


# ==========================================
# SELESAI
# ==========================================