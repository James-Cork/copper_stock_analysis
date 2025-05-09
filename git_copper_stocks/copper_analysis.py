import pandas as pd

#Creating a dataframe using the CSV containing the stock data then using this dataframe to create a dataframe containing only copper price and volume 
dataframe = pd.read_csv('C:/Users/james/MyGIT/git_copper_stocks/US_Stock_Data.csv')
data = {'Copper_Price': dataframe['Copper_Price'], 'Copper_Vol.': dataframe['Copper_Vol.']}
df = pd.DataFrame(data)

#Dropping null values
df = df.dropna()

#Checking for values greater than the mean of a column plus three times the standard deviation as these are likely to be outliers
price_mean = df['Copper_Price'].mean()
price_std = df ['Copper_Price'].std()
price_upper = price_mean + 3*price_std
df = df.drop(df[df['Copper_Price'] > price_upper].index)

volume_mean = df['Copper_Vol.'].mean()
volume_std = df['Copper_Vol.'].std()
volume_upper = volume_mean + 3*volume_std
df = df.drop(df[df['Copper_Vol.'] > volume_upper].index)

#Scaling the remaining data using the MinMaxScaler from sklearn as the copper volume has a much greater range than the price and would therefore impact
#clustering much more than intended if it was not scaled
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
df['Copper_Price'] = sc.fit_transform(df[['Copper_Price']])
df['Copper_Vol.'] = sc.fit_transform(df[['Copper_Vol.']])

#Creating a plot to display the copper price vs volume after wrangling
import matplotlib.pyplot as plt

plt.scatter(x=df['Copper_Price'], y=df['Copper_Vol.'])
plt.title('Copper_Price vs Copper_Vol.')
plt.xlabel('Copper_Price')
plt.ylabel('Copper_Vol.')
plt.show()

#Performing k-means clustering choosing k=4
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4).fit(df[['Copper_Price', 'Copper_Vol.']])

#Printing the cluster centers location and respective index so that each cluster can be identified when interpreting kmeans.labels_
for i in range(len(kmeans.cluster_centers_)):
    print(f'Cluster center {i} located at: {kmeans.cluster_centers_[i]}')

#Counting the number of datapoints in each cluster for use in analysis later
zero_counter = 0
one_counter = 0
two_counter = 0
three_counter = 0
for elem in kmeans.labels_:
    if elem == 0:
        zero_counter += 1
    elif elem == 1:
        one_counter += 1
    elif elem == 2:
        two_counter += 1
    elif elem == 3:
        three_counter += 1

print(f'Cluster 0 Size: {zero_counter}, Cluster 1 Size: {one_counter}, Cluster 2 Size: {two_counter}, Cluster 3 Size: {three_counter}')

#Plotting the copper price vs volume alongside the groups chosen by k-means clustering and the cluster centers for analysis
plt.scatter(x=df['Copper_Price'], y=df['Copper_Vol.'], c=kmeans.labels_)
plt.plot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 'k*', markersize=20)
plt.xlabel('Copper_Price')
plt.ylabel('Copper_Vol.')
plt.show()