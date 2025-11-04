"""Training script for air quality prediction model."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .data_utils import generate_synthetic_data, prepare_data
from .model import AirQualityModel


def train_model(
    model: nn.Module,
    train_loader,
    test_loader,
    epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = "cpu"
) -> Dict[str, List[float]]:
    """
    Train the air quality prediction model.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'test_loss': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Evaluation phase
        model.eval()
        test_losses = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                test_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_test_loss = np.mean(test_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {avg_test_loss:.4f}")
    
    return history


def generate_predictions(
    model: nn.Module,
    test_loader,
    device: str = "cpu",
    n_samples: int = 100
) -> Dict[str, np.ndarray]:
    """
    Generate predictions for visualization.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run inference on
        n_samples: Number of samples to generate predictions for
        
    Returns:
        Dictionary with actual and predicted values
    """
    model.eval()
    model = model.to(device)
    
    actuals = []
    predictions = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features)
            
            actuals.extend(targets.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())
            
            if len(actuals) >= n_samples:
                break
    
    actuals = np.array(actuals[:n_samples]).flatten()
    predictions = np.array(predictions[:n_samples]).flatten()
    
    return {
        'actual': actuals,
        'predicted': predictions
    }


def save_plots(history: Dict, predictions: Dict, output_dir: Path):
    """
    Save training plots.
    
    Args:
        history: Training history
        predictions: Prediction results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'training_history.png')
    plt.close()
    
    # Plot predictions vs actuals
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions['actual'], predictions['predicted'], alpha=0.5)
    min_val = min(predictions['actual'].min(), predictions['predicted'].min())
    max_val = max(predictions['actual'].max(), predictions['predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual PM2.5')
    plt.ylabel('Predicted PM2.5')
    plt.title('Predictions vs Actuals')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'predictions.png')
    plt.close()


def main():
    """Main training function."""
    print("Starting Air Quality Prediction Training...")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create directories
    base_dir = Path(__file__).parent.parent.parent
    models_dir = base_dir / "models"
    outputs_dir = base_dir / "outputs"
    models_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    
    # Generate and prepare data
    print("Generating synthetic data...")
    df = generate_synthetic_data(n_samples=1000)
    train_loader, test_loader = prepare_data(df)
    
    # Create model
    print("Creating model...")
    model = AirQualityModel(input_size=5, hidden_size=64, output_size=1)
    
    # Train model
    print("Training model...")
    history = train_model(
        model, train_loader, test_loader,
        epochs=100, learning_rate=0.001, device=device
    )
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"air_quality_model_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = generate_predictions(model, test_loader, device)
    
    # Save plots
    print("Saving plots...")
    save_plots(history, predictions, outputs_dir)
    
    # Save results metadata
    results = {
        'timestamp': timestamp,
        'final_train_loss': float(history['train_loss'][-1]),
        'final_test_loss': float(history['test_loss'][-1]),
        'device': device,
        'model_path': str(model_path),
        'predictions': {
            'mean_actual': float(predictions['actual'].mean()),
            'mean_predicted': float(predictions['predicted'].mean()),
            'mae': float(np.mean(np.abs(predictions['actual'] - predictions['predicted'])))
        }
    }
    
    results_path = outputs_dir / 'latest_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    print("Training completed successfully!")
    
    return results


if __name__ == "__main__":
    main()
