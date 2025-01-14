import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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