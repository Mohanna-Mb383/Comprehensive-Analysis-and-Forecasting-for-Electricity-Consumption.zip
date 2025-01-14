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
