"""Utilities for generating and processing air quality data."""

import numpy as np
import pandas as pd
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader


class AirQualityDataset(Dataset):
    """Custom dataset for air quality data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            features: Input features
            targets: Target values
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset."""
        return self.features[idx], self.targets[idx]


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic air quality data.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic air quality data
    """
    np.random.seed(seed)
    
    # Generate features
    temperature = np.random.uniform(10, 35, n_samples)  # Celsius
    humidity = np.random.uniform(30, 90, n_samples)  # Percentage
    wind_speed = np.random.uniform(0, 20, n_samples)  # km/h
    pressure = np.random.uniform(980, 1030, n_samples)  # hPa
    traffic_density = np.random.uniform(0, 100, n_samples)  # Arbitrary units
    
    # Generate target (PM2.5 concentration) with some realistic correlations
    pm25 = (
        30 +  # Base level
        0.5 * temperature +  # Higher temperature increases PM2.5
        -0.3 * humidity +  # Higher humidity reduces PM2.5
        -0.8 * wind_speed +  # Wind disperses pollutants
        0.02 * (pressure - 1000) +  # Pressure effect
        0.4 * traffic_density +  # Traffic increases pollution
        np.random.normal(0, 5, n_samples)  # Random noise
    )
    
    # Ensure PM2.5 values are non-negative
    pm25 = np.maximum(pm25, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': pressure,
        'traffic_density': traffic_density,
        'pm25': pm25
    })
    
    return df


def prepare_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare data for training.
    
    Args:
        df: DataFrame with air quality data
        test_size: Proportion of data to use for testing
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Split features and target
    feature_cols = ['temperature', 'humidity', 'wind_speed', 'pressure', 'traffic_density']
    X = df[feature_cols].values
    y = df['pm25'].values.reshape(-1, 1)
    
    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / X_std
    
    # Split data
    split_idx = int(len(X_normalized) * (1 - test_size))
    X_train, X_test = X_normalized[:split_idx], X_normalized[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create datasets
    train_dataset = AirQualityDataset(X_train, y_train)
    test_dataset = AirQualityDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader
