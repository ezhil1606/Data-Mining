import matplotlib.pyplot as plt
import random
import numpy as np

# Plotting histogram
randomlist = [random.randint(0, 1000) for _ in range(100)]  # Assuming you have a list called randomlist
n, bins, patches = plt.hist(randomlist, edgecolor='black')
plt.show()

# Equal-frequency binning
def equalObs(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1), np.arange(nlen), np.sort(x)) #create histogram with equal-frequency bins

n, bins, patches = plt.hist(randomlist, equalObs(randomlist, 10), edgecolor='black')  # Corrected typo here from equal0bs to equalObs
plt.show()

# Random sampling
strata = ["low", "medium", "high"]
randomlist2 = [random.randint(100, 1000) for _ in range(5)]  # Generating a list of 5 random integers between 100 and 1000

print(random.choices(randomlist2, k=5))  # Sampling with replacement
print(random.sample(randomlist2, 5))      # Sampling without replacement

print(np.random.choice(["low", "medium", "high"], size=5, p=[0.3, 0.4, 0.3]))  # Categorical sampling

result_array = np.array(['high', 'high', 'low', 'medium', 'medium'], dtype='<U6')
print(result_array)


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Load crx.data into a dataframe
df = pd.read_csv('crx.data', header=None)

# Change the column names to A1 to A16
df.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']

# Replace all '?' marks with np.nan
df.replace('?', np.nan, inplace=True)

# Convert A2 and A14 attributes to float data type
df['A2'] = pd.to_numeric(df['A2'])
df['A14'] = pd.to_numeric(df['A14'])

# Convert '+' to 1 and "-" to 0 of A16 attribute
df['A16'] = df['A16'].replace({'+': 1, '-': 0})

# Replace values of "A3, A8, A9, A10" attributes to np.nan in 50 random objects
np.random.seed(42) # for reproducibility
random_indices = np.random.choice(df.index, size=50, replace=False)
columns_to_replace = ['A3', 'A8', 'A9', 'A10']
df.loc[random_indices, columns_to_replace] = np.nan

# Save the file as Transformed_crx.csv
df.to_csv('Transformed_crx.csv', index=False)

# Load transformed data
df1 = pd.read_csv('Transformed_crx.csv')

# Missing Values Count After Transformation
print("Missing Values Count After Transformation:")
print(df1.isnull().sum().sort_values(ascending=True))

# Drop rows with missing values
df2 = df1.dropna()

# Impute missing values with mean, median, and mode
imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_median = SimpleImputer(missing_values=np.nan, strategy='median')
imputer_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

dfMean = imputer_mean.fit_transform(df1[['A2', 'A3', 'A8', 'A11', 'A15']])
dfMedian = imputer_median.fit_transform(df1[['A2', 'A3', 'A8', 'A11', 'A15']])
dfMode = imputer_mode.fit_transform(df1[['A2', 'A3', 'A8', 'A11', 'A15']])

# Print missing values count after imputation (Mode)
print("Missing Values Count After Imputation (Mode):")
print(pd.DataFrame(dfMode, columns=['A2', 'A3', 'A8', 'A11', 'A15']).isnull().sum())

# Linear regression imputation
dftestA2 = df1[df1['A2'].isnull()]
xtrain = df2[['A3', 'A8', 'A11', 'A15']]
ytrain = df2['A2']
lr = LinearRegression()
lr.fit(xtrain, ytrain)
xtest = dftestA2[['A3', 'A8', 'A11', 'A15']]
if not xtest.empty:
    dftestA2['A2'] = lr.predict(xtest)
    print("Attributes 'A3', 'A8', 'A11', 'A15' have no NaN values")
print(df1.isnull().sum())
