import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('banking.csv')
df = pd.DataFrame(data)

# Function to transform marital status
def transformMarital(column, value):
    df[column] = np.where(df[column].str.contains(value), 0, 1)

# Applying transformations
transformMarital("marital", "single")
df['housing'] = df['housing'].map({'no': 0, 'yes': 1})
df['loan'] = df['loan'].replace({'no': 0, 'yes': 1})
df['job'] = df['job'].replace({
    'unknown': np.nan, 'management': 0, 'technician': 1, 'entrepreneur': 2,
    'blue-collar': 3, 'retired': 4, 'admin.': 5, 'services': 6,
    'self-employed': 7, 'unemployed': 8, 'housemaid': 9, 'student': 10
})
df['education'] = df['education'].replace({'unknown': np.nan, 'tertiary': 0, 'secondary': 1, 'primary': 2})
df['default'] = df['default'].replace({'yes': 1, 'no': 0})
df['contact'] = df['contact'].replace({'unknown': np.nan, 'telephone': 0, 'cellular': 1})
df['month'] = df['month'].replace({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})
df['poutcome'] = df['poutcome'].replace({'unknown': np.nan, 'failure': 0, 'other': 1, 'success': 2})
df['y'] = df['y'].replace({'no': 0, 'yes': 1})

# Save the processed data
df.to_csv('processed_ban_Q1.csv', index=False)

# Function to apply Z-score normalization
def zscore(df, column):
    mean = np.mean(df[column])
    std_dev = np.std(df[column])
    threshold = 3
    outlier_indices = np.abs((df[column] - mean) / std_dev) > threshold
    df.loc[outlier_indices, column] = np.nan
    return df

# Function to apply Min-Max normalization
def minmax(df, column):
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

# Columns to normalize
columns_to_normalize = ['duration', 'pdays', 'balance']

# Applying Z-score normalization
for col in columns_to_normalize:
    df = zscore(df, col)

# Applying Min-Max normalization
for col in columns_to_normalize:
    df = minmax(df, col)

# Plotting
fig, axs = plt.subplots(len(columns_to_normalize), figsize=(10, 8))
fig.suptitle('Normalized Data')

for i, col in enumerate(columns_to_normalize):
    axs[i].scatter(df.index, df[col], alpha=0.5)
    axs[i].set_title(col)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
