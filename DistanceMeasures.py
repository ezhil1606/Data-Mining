# Calculating Euclidean Distance
# a) General Method or Euclidean Distance

from math import sqrt
import numpy as np

point1 = [1, 3, 5]
point2 = [2, 5, 3]
sqrs = (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2
euc_dist = sqrt(sqrs)
print("Euclidean distance between point1 and point2:", euc_dist)

p1 = np.array((4, 7, 9))
p2 = np.array((10, 12, 14))
euc_dist = np.linalg.norm(p1 - p2)
print("Euclidean distance between p1 and p2:", euc_dist)

# b) Manhattan Distance

def manhattan_distance(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))

print("Manhattan distance between two points:", manhattan_distance([10, 12, 15], [8, 12, 20]))

# c) Minkowski Distance

from decimal import Decimal

def nth_root(value, n_root):
    root_value = 1 / float(n_root)
    return round(Decimal(value) ** Decimal(root_value), 3)

def minkowski_distance(x, y, p_value):
    return nth_root(sum(pow(abs(a - b), p_value) for a, b in zip(x, y)), p_value)

print("Minkowski Distance between two poin ts:", minkowski_distance([0,3, 4, 5], [7, 6, 3, -1], 3))

# d) Cosine Similarity

def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)

def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)

print("Cosine Similarity between two points:", cosine_similarity([3, 45, 7, 2], [2, 54, 13, 15]))

# e) Jaccard similarity

def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)

print("Jaccard Similarity between two points:", jaccard_similarity([0, 1, 2, 5, 6], [0, 2, 3, 5, 7, 9]))

# f) Pearson’s Correlation

# i) Calculating Pearsons Correlation using pandas
import pandas as pd

df = pd.read_excel('Scores.xlsx')
print(df.head())

# Calculating a correlation matrix
print(df.corr())
correlation = df.corr()
print("Correlation between History and English variables:", correlation.loc['History', 'English'])

# ii) Calculating Pearsons Correlation using numpy
import numpy as np

df = pd.read_excel('Scores.xlsx')

corr = np.corrcoef(df['History'], df['English'])
print("Correlation matrix:\n", corr)

# iii) Calculating Pearsons Correlation using scipy
from scipy.stats import pearsonr

# Import your data into Python
df_auto = pd.read_excel('Scores.xlsx')

# Convert dataframe into series
list1 = df_auto['History']
list2 = df_auto['English']

# Apply the pearsonr()
corr, _ = pearsonr(list1, list2)
print('Pearsons correlation: %.3f' % corr)
