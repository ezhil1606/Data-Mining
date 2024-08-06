import pandas as pd
import numpy as np

list_colours=['blue']*3 + ['orange']*2 + ['green']*2
colours=pd.Series(list_colours)
print(colours)

probs = colours.value_counts(normalize=True)
print(probs)

probs_by_hands=[3/7, 2/7, 2/7 ]
print(probs_by_hands)

entropy = -1 * np.sum(np.log2(probs)*probs)
print(entropy)

gini_index = 1 - np.sum(np.square(probs))
print(gini_index)

import pandas as pd
import numpy as np
from collections import Counter
import math

# Function to calculate entropy
def entropy(labels):
    entropy = 0
    label_counts = Counter(labels)
    print("Label counts:", label_counts)
    print("==================================")
    for label in label_counts:
        print("Label:", label)
        prob_of_label = label_counts[label] / len(labels)
        print("Probability of", label, "is", prob_of_label)
        entropy -= prob_of_label * math.log2(prob_of_label)
        print("Entropy of", label_counts, "is", entropy)
        print("==================================")
    return entropy

# Function to calculate information gain
def information_gain(starting_labels, split_labels):
    info_gain = entropy(starting_labels)
    for branched_subset in split_labels:
        info_gain -= len(branched_subset) * entropy(branched_subset) / len(starting_labels)
    print("Information Gain:", info_gain)
    print("====================================")
    return info_gain

# Function to calculate split information
def split_info_calculators():
    diff_labels = df_iris['Species'].value_counts()
    diff_labels = diff_labels / len(df_iris['Species'])
    split_info = -1 * np.sum(np.log2(diff_labels) * diff_labels)
    print("Split Information:", split_info)
    return split_info

# Function to calculate Gini impurity
def gini_impurity(y):
    if isinstance(y, pd.Series):
        p = y.value_counts() / y.shape[0]
        gini = 1 - np.sum(p ** 2)
    return gini

# Function to calculate Gini index
def gini_index_value(starting_labels, split_labels):
    gini_index = gini_impurity(starting_labels)
    for branched_subset in split_labels:
        gini_index += (len(branched_subset) / len(starting_labels)) * gini_impurity(branched_subset)
    print("Gini Index:", gini_index)
    print("====================================")
    return 1 - gini_index

# Function to split dataset based on a column
def split(dataset, column):
    split_data = []
    col_vals = dataset[column].unique()
    for col_val in col_vals:
        split_data.append(dataset[dataset[column] == col_val])
    return split_data

# Function to find the best split attribute
def find_best_split(dataset):
    best_gain = 0
    best_gain_ratio = 0
    best_gini = 0
    best_feature_gain = 0
    best_feature_gainratio = 0
    best_feature_gini = 0

    features = list(dataset.columns)
    features.remove('Species')

    for feature in features:
        split_data = split(dataset, feature)
        split_labels = [dataframe['Species'] for dataframe in split_data]

        gain = information_gain(dataset['Species'], split_labels)
        gain_ratio = gain / split_info_calculators()
        gini = gini_index_value(dataset['Species'], split_labels)

        if gain_ratio > best_gain_ratio:
            best_gain_ratio, best_feature_gainratio = gain_ratio, feature
        if gain > best_gain:
            best_gain, best_feature_gain = gain, feature
        if gini > best_gini:
            best_gini, best_feature_gini = gini, feature

    print("Best Splitting Attribute from gain:", best_feature_gain)
    print("Best Splitting Attribute from gain ratio:", best_feature_gainratio)
    print("Best Splitting Attribute from gini index:", best_feature_gini)
    print("Best Information gain:", best_gain)
    print("Best Gain Ratio:", best_gain_ratio)
    print("Best Gini Index:", best_gini)
    return best_feature_gain, best_gain

# Load iris dataset
df_iris = pd.read_csv("iris.csv")
print('We have {} features in our data'.format(len(df_iris.columns)-1))

# Evaluate features for iris dataset
features = list(df_iris.columns)
features.remove('Species')
def find_best_split(dataset, target_column):
    best_gain = 0
    best_gain_ratio = 0
    best_gini = 0
    best_feature_gain = None
    best_feature_gainratio = None
    best_feature_gini = None

    features = list(dataset.columns)
    if target_column in features:
        features.remove(target_column)

    for feature in features:
        split_data = split(dataset, feature)
        split_labels = [dataframe[target_column] for dataframe in split_data]

        gain = information_gain(dataset[target_column], split_labels)
        gain_ratio = gain / split_info_calculators()
        gini = gini_index_value(dataset[target_column], split_labels)

        if gain_ratio > best_gain_ratio:
            best_gain_ratio, best_feature_gainratio = gain_ratio, feature
        if gain > best_gain:
            best_gain, best_feature_gain = gain, feature
        if gini > best_gini:
            best_gini, best_feature_gini = gini, feature

    print("Best Splitting Attribute from gain:", best_feature_gain)
    print("Best Splitting Attribute from gain ratio:", best_feature_gainratio)
    print("Best Splitting Attribute from gini index:", best_feature_gini)
    print("Best Information gain:", best_gain)
    print("Best Gain Ratio:", best_gain_ratio)
    print("Best Gini Index:", best_gini)
    return best_feature_gain, best_gain

best_feature, best_gain = find_best_split(df_iris, 'Species')
new_data = split(df_iris, best_feature)

# Load tennis dataset
df_tennis = pd.read_csv("tennis.csv")
features = list(df_tennis.columns)
features.remove('play')
# Function to find the best split attribute
def find_best_split(dataset, target_column):
    best_gain = 0
    best_gain_ratio = 0
    best_gini = 0
    best_feature_gain = None
    best_feature_gainratio = None
    best_feature_gini = None

    features = list(dataset.columns)
    if target_column in features:
        features.remove(target_column)

    for feature in features:
        split_data = split(dataset, feature)
        split_labels = [dataframe[target_column] for dataframe in split_data]

        gain = information_gain(dataset[target_column], split_labels)
        gain_ratio = gain / split_info_calculators()
        gini = gini_index_value(dataset[target_column], split_labels)

        if gain_ratio > best_gain_ratio:
            best_gain_ratio, best_feature_gainratio = gain_ratio, feature
        if gain > best_gain:
            best_gain, best_feature_gain = gain, feature
        if gini > best_gini:
            best_gini, best_feature_gini = gini, feature

    print("Best Splitting Attribute from gain:", best_feature_gain)
    print("Best Splitting Attribute from gain ratio:", best_feature_gainratio)
    print("Best Splitting Attribute from gini index:", best_feature_gini)
    print("Best Information gain:", best_gain)
    print("Best Gain Ratio:", best_gain_ratio)
    print("Best Gini Index:", best_gini)
    return best_feature_gain, best_gain

# Find best split for tennis dataset
best_feature, best_gain = find_best_split(df_tennis, 'play')
new_data = split(df_tennis, best_feature)
