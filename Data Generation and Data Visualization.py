import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv("data.csv")
df = pd.DataFrame(data)

plt.title("Game Global Sales - Top Ten")
plt.xlabel("Game Rank")
plt.ylabel("Game Global Sales")
x = np.array(df['Rank'].iloc[0:10])
y = np.array(df['Global_Sales'].iloc[0:10])
plt.plot(x, y)
plt.show()

plt.title("Game Global Sales - Top Ten")
plt.xlabel("Game Rank")
plt.ylabel("Game Global sales")
x = np.array(df['Rank'].iloc[0:10])
y = np.array(df['Global_Sales'].iloc[0:10])
plt.plot(x, y, color='red', linestyle='--', linewidth=5)
plt.legend(['Rank Vs Sales'])
plt.show()

plt.subplots(figsize=(10, 6))
plt.title("Game Global Sales - Top Ten")
plt.xlabel("Game Rank")
plt.ylabel("Game Global sales")
x = np.array(df['Rank'].iloc[0:10])
y = np.array(df['Global_Sales'].iloc[0:10])
plt.scatter(x, y, color='red')
plt.show()



plt.subplots(figsize=(10, 6))

df1=df.groupby(['Year'], as_index=False)['Global_Sales' ].sum()
plt.title("Global Sales by year")

plt.xlabel("Year")

plt.ylabel("Global sales")
df1=df1.set_index('Year')

df1['Global_Sales'].plot(kind='bar', color='brown')

plt.subplots(figsize=(10, 6))

df1=df.groupby(['Platform'], as_index=False)['Global_Sales'].sum()
plt.title("Global Sales by Platform")

plt.xlabel("Platform")

plt.ylabel("Global sales")

df1=df1.set_index('Platform')
df1['Global_Sales'].plot(kind='bar', color='brown')
plt.subplots(figsize=(10, 6))

df1=df.groupby(['Platform'], as_index=False)['Global_Sales'].sum()
plt.title("Global Sales by Platform")

plt.xlabel("Platform")

plt.ylabel("Global sales")

df1=df1.sort_values(by=['Global_Sales'], ascending=False)
df1=df1.set_index('Platform')
df1['Global_Sales'].plot(kind='bar', color='brown')
plt.show()

df2 = df[['Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales']]
df2 = df2.groupby(['Genre'], as_index=False)[['NA_Sales', 'EU_Sales', 'JP_Sales']].sum()
df2.plot(kind='bar', stacked=True, figsize=(12, 6), color=['red', 'blue', 'green'])

from sklearn import preprocessing
plt.subplots(figsize=(10, 6))

label_encoder = preprocessing.LabelEncoder()

df['genre_enc']= label_encoder.fit_transform(df['Genre'])
df41=df.set_index('genre_enc')
df42 = df41.groupby(df41.index, as_index=False)[['Name']].count()
plt.hist(data=data,x='Genre')

plt.title("Game Titles by Genre")
plt. show()

fig=plt.figure(figsize=(12,6))
plt.subplot(231)
plt.title('NA_Sales')
plt.hist(data=data,x='NA_Sales')
plt.subplot(2,3,2)
plt.title("EU_Sales")
plt.hist(data=data,x='EU_Sales')
plt.subplot(2,3,4)
plt.title("JP_Sales")
plt.hist(data=data,x="JP_Sales")
plt.subplot(2,3,5)
plt.title("Global Sales")
plt.hist(data=data,x='Global_Sales')
df.head()


import matplotlib.pyplot as plt

x1 = df['NA_Sales']
x2 = df['EU_Sales']
x3 = df['JP_Sales']
x4 = df['Other_Sales']

x = [x1, x2, x3, x4]
plt.figure(figsize=(8, 6))
plt.boxplot(x)
plt.title('Sales Distribution by Region')
plt.xlabel('Region')
plt.ylabel('Sales')
plt.xticks([1, 2, 3, 4], ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])
plt.show()


plt.figure(figsize=(12, 6))
plt.scatter(x=df['Year'], y=df['Global_Sales'])
plt.xlabel('Year')
plt.ylabel('Global Sales')
plt.title('Global Sales by Year')
plt.show()

plt.figure(figsize=(12, 6))
sales_by_genre = df.groupby('Genre').sum()['NA_Sales']
plt.pie(sales_by_genre, labels=sales_by_genre.index)
plt.title('NA Sales by Genre')
plt.axis('equal')
plt.show()

# Q6
from sklearn.datasets import make_regression
from matplotlib import pyplot
fig=plt.figure(figsize=(12,6))
X_test, y_test = make_regression(n_samples=150, n_features=1, noise=5)
plt.scatter(X_test,y_test)
from sklearn.datasets import make_classification
fig=plt.figure(figsize=(12,6))
X1, Y1 = make_classification(
n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")
from sklearn.datasets import make_blobs
fig=plt.figure(figsize=(12,6))
X1, Y1 = make_blobs(n_features=2, centers=3)
plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")
from sklearn.datasets import make_circles
fig=plt.figure(figsize=(12,6))
X1,Y1=make_circles(n_samples=100,factor=0.8)
pyplot.scatter(X1[:, 0],X1[:,1],marker='o',c=Y1)
from sklearn.datasets import make_moons
fig=plt.figure(figsize=(12,6))
X1,Y1=make_moons(n_samples=100)
pyplot.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")

from faker import Factory
import pandas as pd
import random
import numpy as np


def create_fake_cust(fake):
    cust = [
        random.randint(100, 1000000),
        fake.name(),
        fake.address(),
        fake.date_time(),
        np.random.choice(["M", "F"], p=[0.5, 0.5]),
        np.random.choice(["O+", "O-", "AB+", "AB-", "A+", "A-", "B+", "B-"],
                         p=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]),
        fake.phone_number(),
        random.randint(1000, 2000)
    ]
    return cust


if __name__ == '__main__':
    fake = Factory.create()
    df = pd.DataFrame(columns=['ID', 'Name', 'Address', 'Date', 'Gender', 'Blood_Group', 'Phone_Number', 'Random_Num'])

    for _ in range(100):
        df.loc[len(df)] = create_fake_cust(fake)
        df.to_csv('customer.csv', index=False)
