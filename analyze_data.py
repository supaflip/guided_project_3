import pandas as pd 
import numpy as np 
import seaborn as sns 
import pickle
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split



df = pd.read_csv("./data/troop_movements.csv")
sns.set()

# print(df.describe())
# print(df[:5])
# print(df.dtypes)

# Create grouped data
def get_counts(df, col):
    count = pd.DataFrame(df[col].value_counts())
    # print(count)
    # print("\n")
    return count

# print(map(lambda x: get_counts(df, x), ["empire_or_resistance", "homeworld", "unit_type"]))

cols = ["empire_or_resistance", "homeworld", "unit_type"]

for c in cols:
    get_counts(df, c)

# Engineer new feature called is_resistance with a True or False value
df["is_resistance"] = (df["empire_or_resistance"] == "resistance")
# print(df["is_resistance"])

#Create a bar plot showing Empire vs Resistance distribution
def graph_counts(df, col):
    count = get_counts(df, col)
    sns.barplot(count.T)
    plt.show()

cols2 = ["empire_or_resistance"]

for c in cols2:
    graph_counts(df, c)


# Create prediction model that predicts if character is joining empire or resistance based on homeworld and unit_type

X = df[["homeworld", "unit_type"]]
y = df["is_resistance"] # boolean in numerical represenation (0, 1)

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
pickle.dump(model, open("decision_tree.pkl", "wb")) # save the model as a pickle file

y_pred = model.predict(X_test)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.show()

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame to hold the feature importances
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
# print(feature_importances)

# Sorting by highest to lowest Importance
sorted_importances = feature_importances.sort_values(by = "Importance", ascending = False)

# Create a bar plot that shows feature importance
plt.figure(figsize=(8, 6))
plt.bar(sorted_importances["Feature"], sorted_importances["Importance"])
plt.show()





