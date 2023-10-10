import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 


df = pd.read_csv("./data/troop_movements.csv")
sns.set()

# print(df.describe())
# print(df[:5])
# print(df.dtypes)

# Create grouped data
def get_counts(df, col):
    count = pd.DataFrame(df[col].value_counts())
    print(count)
    print("\n")
    return count

# print(map(lambda x: get_counts(df, x), ["empire_or_resistance", "homeworld", "unit_type"]))

cols = ["empire_or_resistance", "homeworld", "unit_type"]

for c in cols:
    get_counts(df, c)

# Engineer new feature called is_resistance with a True or False value
df["is_resistance"] = (df["empire_or_resistance"] == "resistance")
print(df["is_resistance"])

#Create a bar plot showing Empire vs Resistance distribution
def graph_counts(df, col):
    count = get_counts(df, col)
    sns.barplot(count.T)
    plt.show()

cols2 = ["empire_or_resistance"]

for c in cols2:
    graph_counts(df, c)
