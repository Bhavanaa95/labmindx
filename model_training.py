import pandas as pd

df = pd.read_csv("Titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin"])

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"])

df = df.drop(columns=["Name", "Ticket", "PassengerId"])

y = df["Survived"]
X = df.drop("Survived", axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, class_weight="balanced")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix, classification_report

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import pandas as pd

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.coef_[0]
})

feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)

import matplotlib.pyplot as plt
plt.figure()

plt.barh(feature_importance["Feature"], feature_importance["Importance"])

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")

plt.savefig("feature_importance.png")

plt.show()

colors = ["green" if val > 0 else "red" for val in feature_importance["Importance"]]

plt.barh(feature_importance["Feature"], feature_importance["Importance"], color=colors)