#!/usr/bin/env python
# coding: utf-8

# In[80]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


# In[81]:


# Adding datasheet to the file
gt= pd.read_csv('2021.Vans_Aggregated.csv')
gt.dropna()


# In[82]:


gt.describe()


# In[83]:


# Exclude non-numeric columns
numeric_data = gt.select_dtypes(include=[np.number])

# Correlation matrix
corr_matrix = numeric_data.corr()
corr_matrix


# In[84]:


# Kurtosis
kurtosis = numeric_data.kurtosis()
print("\nKurtosis:")
print(kurtosis)


# In[85]:


# Skewness
skewness = numeric_data.skew()
print("\nSkewness:")
print(skewness)


# In[86]:


def create_relational_graph(x, y):
    """
    Create a scatter plot to visualize the relationship between two variables.

    Parameters:
    x (array-like): gt for the x-axis.
    y (array-like): gt for the y-axis.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='red', alpha=0.5)
    plt.xlabel('OBFCM Fuel consumption (l/100 km)')
    plt.ylabel('WLTP Fuel consumption (l/100 km)')
    plt.title('Relationship Between OBFCM Fuel consumption (l/100 km) and WLTP Fuel consumption (l/100 km)')
    plt.grid(True)
    plt.show()


create_relational_graph(gt['OBFCM Fuel consumption (l/100 km)'], gt['WLTP Fuel consumption (l/100 km)'])


# In[87]:


def plot_column_bar(data, column_name):
    """
    Create a bar graph to visualize the distribution of a specified categorical column in the dataset.

    Parameters:
    data (DataFrame): Pandas DataFrame containing the dataset.
    column_name (str): Name of the categorical column to visualize.

    Returns:
    None
    """
    # Count the frequency of each category
    category_counts = data[column_name].value_counts()

    # Plot the bar graph
    plt.figure(figsize=(8, 6))
    category_counts.plot(kind='bar', color='yellow')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title(f'Bar Graph of {column_name}')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()
plot_column_bar(gt, 'Fuel Type')


# In[88]:


import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap_subset(data, start_column, end_column):
    """
    Create a heatmap to visualize the correlation matrix of a subset of columns in the data.

    Parameters:
    data (DataFrame): The input data.
    start_column (int): Index of the starting column.
    end_column (int): Index of the ending column.

    Returns:
    None
    """
    # Select the subset of columns
    subset_data = data.iloc[:, start_column:end_column+1]

    # Filter out non-numeric columns
    numeric_subset_data = subset_data.select_dtypes(include=['float64', 'int64'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_subset_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap for Columns {} to {}'.format(start_column, end_column))
    plt.show()

# Example usage for columns 2 to 6
plot_heatmap_subset(gt, 2, 6)


# In[89]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

data = pd.read_csv('2021.Vans_Aggregated.csv')

# Global variable for scaler
scaler = StandardScaler()

def preprocess_data(df):
    """
    Preprocess the data by handling missing values and selecting appropriate columns for clustering.

    Parameters:
        df: DataFrame containing the data.

    Returns:
        DataFrame: Preprocessed data.
    """
    # Select appropriate columns for clustering
    columns_for_clustering = ['Number of vehicles','OBFCM Fuel consumption (l/100 km)','WLTP Fuel consumption (l/100 km)','absolute gap Fuel consumption (l/100 km)','percentage gap Fuel consumption (%)']

    # Handle missing values
    df.dropna(subset=columns_for_clustering, inplace=True)
    
    return df[columns_for_clustering]

def performing_elbow_method(data_scaled):
    """
    Calculate the optimal number of clusters using the elbow method.

    Parameters:
        data_scaled: Rescaled data.

    Returns:
        integer: Optimal number of clusters.
    """
    # within cluster sum of squares
    wcss = []
    # Adjusted range to avoid ValueError
    for i in range(1, 6):
        kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)  # Explicitly setting n_init
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)
    # Plotting the within-cluster sum of squares
    plt.plot(range(1, 6), wcss, marker='o')
    # Adding labels and title to the plot
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    # Adding grid
    plt.grid(True)
    plt.show()
    return 3

def visualize_clusters(data_scaled, optimal_num_clusters):
    """
    Visualizing the clusters based on the number of clusters.

    Parameters:
        data_scaled: Rescaled data.
        optimal_num_clusters: Optimal number of clusters.
    """
    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(data_scaled)
    cluster_centers = kmeans.cluster_centers_
    # Plotting the clusters
    plt.figure(figsize=(8, 6))
    for cluster_label in range(optimal_num_clusters):
        plt.scatter(data_scaled[cluster_labels == cluster_label, 0],
                    data_scaled[cluster_labels == cluster_label, 1],
                    label=f'Cluster {cluster_label + 1}')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='blue', label='Cluster Centers')
    # Adding labels and title to the plot
    plt.xlabel('OBFCM Fuel consumption (l/100 km)')
    plt.ylabel('WLTP Fuel consumption (l/100 km)')
    plt.title('KMeans Clustering')
    plt.legend()
    # Adding grid
    plt.grid(True)
    plt.show()
    
    
def evaluate_clustering_accuracy(data_scaled, optimal_num_clusters):
    """
    Evaluate the accuracy of clustering predictions using silhouette score.

    Parameters:
        data_scaled: Rescaled data.
        optimal_num_clusters: Optimal number of clusters.

    Returns:
        float: Silhouette score.
    """
    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(data_scaled)
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    return silhouette_avg

# Preprocessing the data
data_processed = preprocess_data(data)

# Standardizing the data
data_scaled = scaler.fit_transform(data_processed)


# Performing the elbow method to determine the optimal number of clusters
optimal_num_clusters = performing_elbow_method(data_scaled)

# Visualizing the clusters
visualize_clusters(data_scaled, optimal_num_clusters)



# Evaluating clustering accuracy
silhouette_accuracy = evaluate_clustering_accuracy(data_scaled, optimal_num_clusters)
print("Silhouette Score:", silhouette_accuracy)


# In[90]:


from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
data = gt.head(n=40)
def line_fit_and_plot(data):
    """
    Fit a linear regression model to the provided data and plot the results.

    Parameters:
        data: DataFrame containing appropriate columns.
    """
    X = data['OBFCM CO2 emissions (g/km)'].values.reshape(-1)  # Independent variable
    y = data['WLTP CO2 emissions (g/km)'].values  # Dependent variable

    # Fitting the linear regression model
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)

    # Predictions
    y_pred = model.predict(X.reshape(-1, 1))

    # Confidence interval and error bars
    confidence = 0.95
    n = len(y)
    mse = np.mean((y - y_pred) ** 2)
    alpha = 1 - confidence
    t_critical = np.abs(stats.t.ppf(alpha / 2, df=n-1))
    margin_of_error = t_critical * np.sqrt((mse / n) * (1 + (1/n) + ((X - X.mean()) ** 2) / np.sum((X - X.mean()) ** 2)))
    confidence_interval = np.array([y_pred - margin_of_error, y_pred + margin_of_error])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='yellow', label='Actual Data')
    plt.plot(X, y_pred, color='brown', label='Linear Fit')
    plt.fill_between(X, confidence_interval[0], confidence_interval[1], color='orange', alpha=0.3, label='95% Confidence Interval')
    plt.errorbar(X, y_pred, yerr=margin_of_error, fmt='o', color='black', label='Error Bars')

    # Labels and title
    plt.xlabel('OBFCM CO2 emissions (g/km)')
    plt.ylabel('WLTP CO2 emissions (g/km)')
    plt.title('Linear Regression and Confidence Interval')
    plt.legend()
    plt.grid(True)
    plt.show()

# Perform line fitting and plot
line_fit_and_plot(data)


# In[91]:


# Preprocess the data
data_processed = preprocess_data(data)

# Define features (X) and target (y)
X = data_processed[['Number of vehicles','OBFCM Fuel consumption (l/100 km)','WLTP Fuel consumption (l/100 km)','absolute gap Fuel consumption (l/100 km)','percentage gap Fuel consumption (%)']]
y = data_processed['Number of vehicles']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Visualize the fitting predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='pink')  # Identity line
plt.xlabel('OBFCM Fuel consumption (l/100 km)')
plt.ylabel('WLTP Fuel consumption (l/100 km)')
plt.title('Fitting Predictions')
plt.grid(True)
plt.show()

print("Mean Squared Error:", mse)


# In[ ]:




