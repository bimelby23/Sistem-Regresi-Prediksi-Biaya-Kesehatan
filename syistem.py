import pandas as pd

df = pd.read_csv('insurance.csv')

print(df.head())

print("\nInfo Data:")
print(df.info())

print("\nShape Data:", df.shape)
