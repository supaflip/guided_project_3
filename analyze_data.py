import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
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
y = df["is_resistance"] # boolean

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)





