import pandas as pd
df = pd.read_csv("Titanic.csv")

print(df.head())

print("Shape: " , df.shape)

df.info()

print(df.isnull().sum())

print(df.dtypes)

print(df["Survived"].value_counts())

df["Age"] = df["Age"].fillna(df["Age"].median())
                             
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df = df.drop(columns=["Cabin"])

print("\nMissing after cleaning:")
print(df.isnull().sum())

print("\nAge sample:")
print(df["Age"].head(10))

print(df["Sex"].unique())
print(df["Embarked"].unique())

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
print(df["Sex"].head())

print(df["Sex"].dtypes)

df = pd.get_dummies(df, columns=["Embarked"])

print(df.head())
print(df.columns)

df = df.drop(columns=["Name", "Ticket", "PassengerId"])
print(df.columns)

print(df.dtypes)






