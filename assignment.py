# -*- coding: utf-8 -*-
"""code-codementor-1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZtAR_4xj5SRLcL38w8e-ayIuG3vSuB7K

# Task 1
"""

import pandas as pd

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = "./Electricity Consumption.csv"  # Replace with your file path
data = pd.read_csv(file_path)
print (data.head())
postalcode_data = pd.read_csv('./Postal Codes - Lleida.csv')
postalcode = postalcode_data['CODPOS']

import zipfile
with zipfile.ZipFile("/content/Spain_shapefile.zip","r") as zip_ref:
    zip_ref.extractall("./Spain_shapefile")
# !zip -r  ./

# Convert time to datetime and extract date and hour
data['time'] = pd.to_datetime(data['time'])
data['date'] = data['time'].dt.date
data['hour'] = data['time'].dt.hour
print (data.head())

# Pivot to create a matrix of hourly consumption per day per postal code
pivot_data = data.pivot_table(index=['postalcode', 'date'], columns='hour', values='consumption', aggfunc='sum', fill_value=0)
print (pivot_data.head())

# postalcode_grouped = pivot_data.groupby('postalcode').mean()
# print (postalcode_grouped.head())
features_scaled = pivot_data.div(pivot_data.sum(axis=1), axis=0)
# print (features_scaled.head())

# Determine the optimal number of clusters using silhouette scores
range_n_clusters = range(2, 11)
silhouette_scores = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title("Silhouette Scores for Different Cluster Sizes")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

# Choose the optimal number of clusters and fit KMeans
optimal_clusters = range_n_clusters[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(features_scaled)

# Add cluster labels to the data
pivot_data['cluster'] = cluster_labels
pivot_data.to_csv('postalcode_clusters.csv', index=True)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# Add PCA components for visualization
pivot_data['pca1'] = features_pca[:, 0]
pivot_data['pca2'] = features_pca[:, 1]

# Visualize the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='pca1',
    y='pca2',
    hue='cluster',
    palette='Set2',
    data=pivot_data,
    style='cluster'
)
plt.title('Clustering of Typical Daily Electricity Load Curves by Postal Code')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Analyze the cluster centroids to identify common patterns
centroids = kmeans.cluster_centers_
centroid_df = pd.DataFrame(centroids, columns=features_scaled.columns)
print("Cluster Centroids (Typical Daily Patterns):")
print(centroid_df)

"""# Task 2"""



import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
from geopy.geocoders import Nominatim

# Define the UTM coordinate system (EPSG: 25830) and the geographic coordinate system (EPSG: 4326)
utm_proj = "epsg:25830"  # UTM zone 30N (Spain)
latlon_proj = "epsg:4326"  # WGS84 (lat/lon)
transformer = Transformer.from_crs(utm_proj, latlon_proj, always_xy=True)

# Load cadaster data
cadaster_data = pd.read_csv("./Cadaster Lleida.csv")

# Create a GeoDataFrame from cadaster coordinates
geometry = [Point(*transformer.transform(x, y)) for x, y in zip(cadaster_data['X'], cadaster_data['Y'])]
cadaster_gdf = gpd.GeoDataFrame(cadaster_data, geometry=geometry, crs="EPSG:4326")

# Load postal code boundaries (shapefile or GeoJSON)
postal_code_boundaries = gpd.read_file("es_1km.shp")  # Replace with your file path

# Ensure both datasets are in the same coordinate reference system (CRS)
postal_code_boundaries = postal_code_boundaries.to_crs(cadaster_gdf.crs)

# Perform spatial join to assign postal codes to cadaster data
cadaster_with_postal_codes = gpd.sjoin(
    cadaster_gdf,
    postal_code_boundaries,
    how="left",
    predicate="intersects"  # Explicit predicate required in modern GeoPandas
)

# Save or use the resulting GeoDataFrame
cadaster_with_postal_codes.to_csv("cadaster_with_postal_codes.csv", index=False)

# Print the resulting GeoDataFrame
print(cadaster_with_postal_codes)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, log_loss
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load datasets
electricity_data = pd.read_csv("./Electricity Consumption.csv")
cluster_data = pd.read_csv("./postalcode_clusters.csv")  # Cluster labels from previous task
weather_data = pd.read_csv("./Weather.csv")  # Weather: temp, humidity, wind, etc.
# cadaster_data = pd.read_csv("cadaster_data.csv")  # Property density, building types, etc.
socio_economic_data = pd.read_csv("./Socio-Economic.csv")  # Income, population, etc.

postalcode_data = pd.read_csv('./Postal Codes - Lleida.csv')
postalcode = postalcode_data['CODPOS']

# Preprocess electricity data
electricity_data['time'] = pd.to_datetime(electricity_data['time'])
electricity_data['date'] = electricity_data['time'].dt.date

# Aggregate daily consumption
daily_consumption = electricity_data.groupby(['postalcode', 'date'])['consumption'].sum().reset_index()
daily_consumption = daily_consumption.rename(columns={'consumption': 'daily_consumption'})

# Merge weather data (by postal code and date)
weather_data['time'] = pd.to_datetime(weather_data['time'])
weather_data['date'] = weather_data['time'].dt.date
data = daily_consumption.merge(weather_data, on=['postalcode', 'date'], how='left')

data = data.merge(cluster_data, on=['postalcode', 'date'], how='right')

# Extract year from the date column for the merge
data['year'] = pd.to_datetime(data['date']).dt.year

# Merge cadastral and socio-economic data (by postal code)
# data = data.merge(cadaster_data, on='postalcode', how='left')
data = data.merge(socio_economic_data, on=['postalcode', 'year'], how='left')

# Add lagged features (e.g., previous day's consumption)
data['prev_day_consumption'] = data.groupby('postalcode')['daily_consumption'].shift(1)


# Drop rows with missing values
data.dropna(axis = 0)

for code in postalcode:
    print (code)
    # Prepare features and target
    features = ['prev_day_consumption', 'airtemperature', 'relativehumidity', 'windspeed',
                'population', 'peopleperhousehold']  # Add relevant features
    data_new = data[data['postalcode'] == code].copy()
    # print (data_new)
    X = data_new[features]
    y = data_new['cluster']
    # print (y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    # Make predictions and calculate probabilities
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)

    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    logloss = log_loss(y_test, y_prob)
    print(f"Log Loss: {logloss}")

    # Forecast probabilities for the next day
    next_day_features = X_test[:1]  # Replace with actual next day's features
    next_day_probabilities = rf.predict_proba(next_day_features)

    print("Next Day Cluster Probabilities:")
    print(next_day_probabilities)

"""# Task 3"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load and merge datasets (electricity consumption, weather, cadaster, socio-economic)
consumption_data = pd.read_csv("./Electricity Consumption.csv")
weather_data = pd.read_csv("./Weather.csv")
# cadaster_data = pd.read_csv("cadaster_with_postal_codes.csv")
socio_economic_data = pd.read_csv("./Socio-Economic.csv")

postalcode_data = pd.read_csv('./Postal Codes - Lleida.csv')
postalcode = postalcode_data['CODPOS']

# Merge data on common keys like postal codes and timestamps
merged_data = consumption_data.merge(weather_data, on=["postalcode", "time"], how="left")
# merged_data = merged_data.merge(cadaster_data, on="postalcode", how="left")
merged_data = merged_data.merge(socio_economic_data, on="postalcode", how="left")

# Preprocess data
merged_data["time"] = pd.to_datetime(merged_data["time"])
merged_data.set_index("time", inplace=True)
merged_data.sort_index(inplace=True)
merged_data = merged_data.dropna()
# Fill missing values
merged_data.ffill()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

# Function to create sequences for time-series forecasting
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data) - 96):  # 96 steps for 96-hour forecast
        X.append(data[i - lookback:i, :-1])  # Exclude the target column
        y.append(data[i:i + 96, -1])  # Target column for next 96 hours
    return np.array(X), np.array(y)

# Define function to create LSTM model
def create_model(optimizer='adam', dropout_rate=0.2, lstm_units=64):
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(0.01), input_shape=(lookback, len(features))),
        Dropout(dropout_rate),
        LSTM(lstm_units // 2, kernel_regularizer=l2(0.01)),
        Dropout(dropout_rate),
        Dense(96, kernel_regularizer=l2(0.01))  # 96-hour forecast
    ])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

postal_codes = merged_data['postalcode'].unique()

# Parameters
lookback = 48
target = "consumption"
features = ['airtemperature', 'relativehumidity', 'windspeed', 'population']

# Manually define hyperparameters
manual_hyperparameters = {
    'optimizer': 'adam',
    'dropout_rate': 0.2,
    'lstm_units': 128,
    'batch_size': 32,
    'epochs': 20
}

# Iterate through each postal code
for code in postal_codes:
    print(f"Processing postal code: {code}")

    # Filter data for the postal code
    merged_data_new = merged_data[merged_data['postalcode'] == code].copy()
    data = merged_data_new[features + [target]].to_numpy()[:1000]  # Ensure data is numpy array

    # Normalize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Create sequences
    X, y = create_sequences(data_scaled, lookback)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Create and compile model
    model = create_model(
        optimizer=manual_hyperparameters['optimizer'],
        dropout_rate=manual_hyperparameters['dropout_rate'],
        lstm_units=manual_hyperparameters['lstm_units']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Train the model
    history = model.fit(X_train, y_train, validation_split=0.2,
                        epochs=manual_hyperparameters['epochs'],
                        batch_size=manual_hyperparameters['batch_size'],
                        verbose=1,
                        callbacks=[early_stopping, lr_scheduler])

    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    # Forecast
    y_pred = model.predict(X_test)

    # Rescale predictions and ground truth
    y_pred_rescaled = y_pred * scaler.scale_[-1] + scaler.mean_[-1]
    y_test_rescaled = y_test * scaler.scale_[-1] + scaler.mean_[-1]

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_rescaled.flatten(), y_pred_rescaled.flatten()))
    mae = mean_absolute_error(y_test_rescaled.flatten(), y_pred_rescaled.flatten())
    print(f"RMSE: {rmse}, MAE: {mae}")

    # Plot actual vs predicted
    plt.figure(figsize=(15, 5))
    plt.plot(y_test_rescaled.flatten()[:500], label="Actual")
    plt.plot(y_pred_rescaled.flatten()[:500], label="Predicted")
    plt.legend()
    plt.title(f"Actual vs Predicted Consumption for Postal Code {code}")
    plt.show()

!pip install scikeras

