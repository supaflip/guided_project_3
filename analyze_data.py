import pandas as pd 
import numpy as np 

df = pd.read_csv("./data/troop_movements.csv")

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
    get_counts(df,c)

# Engineer new feature called is_resistance with a True or False value
df["is_resistance"] = (df["empire_or_resistance"] == "resistance")
print(df["is_resistance"])