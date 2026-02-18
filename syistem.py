import pandas as pd

df = pd.read_csv('insurance.csv')

print(df.head())

print("\nInfo Data:")
print(df.info())

print("\nShape Data:", df.shape)



print("\nMissing Value:")
print(df.isnull().su
df = df.dropna()

print("\nData Duplikat:", df.duplicated().sum())

df = df.drop_duplicates()




df['sex'] = df['sex'].map({'male':1, 'female':0})
df['smoker'] = df['smoker'].map({'yes':1, 'no':0})
df['region'] = df['region'].map({
    'northeast':0,
    'northwest':1,
    'southeast':2,
    'southwest':3
})


print("\nStatistik Data:")
print(df.describe())

X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']





from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("\nIntercept:", model.intercept_)

print("\nKoefisien:")
for fitur, coef in zip(X.columns, model.coef_):
    print(fitur, ":", coef)

y_pred = model.predict(X_test)

print("\nR2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

