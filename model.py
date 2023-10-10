import pandas as pd 
import numpy as np 
import seaborn as sns 
import pickle
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

#GLOBALS ####################
ENABLE_PRINT = False
SHOW_GRAPHS = True
DATAFILE = "./data/troop_movements.csv"
CLASSIFIER = DecisionTreeClassifier
sns.set()

#FUNCTIONS ####################
def examine_df(df):
    print(df.describe())
    print(df[:5])
    print(df.dtypes)

def apply_to_cols(fn, *cols):
    for c in cols:
        fn(c)

# Create grouped data
def get_counts(df, col, verbose=False):
    count = pd.DataFrame(df[col].value_counts())
    if verbose and ENABLE_PRINT:
        print(count)
        print("\n")
    return count

#Create a bar plot showing value distribution
def graph_counts(df, col):
    count = get_counts(df, col, False)
    sns.barplot(count.T)
    if SHOW_GRAPHS:
        plt.show()

def make_model(model_class, X_train, y_train, seed=True, filename="model.pkl"):
    if seed:
        model = model_class(random_state=42)
    else:
        model = model_class()

    model.fit(X_train, y_train)
    pickle.dump(model, open(filename, "wb"))
 
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    if SHOW_GRAPHS: plt.show()

def graph_weights(model, verbose=False):
    # Get feature importances
    importances = model.feature_importances_

    # Create a DataFrame to hold the feature importances
    feature_importances = pd.DataFrame({'Feature': model.feature_names_in_, 'Importance': importances})
    if verbose and ENABLE_PRINT:
        print (feature_importances)

    # Sorting by highest to lowest Importance
    sorted_importances = feature_importances.sort_values(by = "Importance", ascending = False)

    # Create a bar plot that shows feature importance
    if SHOW_GRAPHS:
        plt.figure(figsize=(8, 6))
        plt.bar(sorted_importances["Feature"], sorted_importances["Importance"])
        plt.show()

#MAIN #################################

def main(datafile, classifier=CLASSIFIER):
    df = pd.read_csv(datafile)

    apply_to_cols(lambda c: get_counts(df, c, True), "empire_or_resistance", "homeworld", "unit_type")

    # Engineer new feature called is_resistance with a True or False value
    df["is_resistance"] = (df["empire_or_resistance"] == "resistance")
    if ENABLE_PRINT: print(df["is_resistance"])

    apply_to_cols(lambda c: graph_counts(df, c), "empire_or_resistance")

    # Create prediction model that predicts if character is joining empire or resistance based on homeworld and unit_type
    X = df[["homeworld", "unit_type"]]
    y = df["is_resistance"]

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if ENABLE_PRINT: print(X_train.shape, y_train.shape)

    pkl_filename = f'{datafile.split("/")[-1].strip(".csv")}_model.pkl'
    model = make_model(classifier, X_train, y_train, filename=pkl_filename)

    evaluate_model(model, X_test, y_test)
    graph_weights(model)

    return model

main(DATAFILE)