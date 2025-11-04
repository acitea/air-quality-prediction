import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def train_model(
    model,
    train_loader,
    test_loader,
) -> Dict[str, List[float]]:
    """
    Train the air quality prediction model.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    Returns:
        Dictionary with training history
    """
    history = {
        'train_loss': [],
        'test_loss': []
    }

    return history


def generate_predictions(
    model,
    n_samples: int = 10
) -> Dict[str, np.ndarray]:
    """
    Generate predictions for visualization.
    
    Args:
        model: Trained model
        n_samples: Number of samples to generate predictions for
        
    Returns:
        Dictionary with actual and predicted values
    """
    
    # TODO
    actuals = []
    predictions = []
    return {
        'actual': actuals,
        'predicted': predictions
    }

# TODO: Implement Plots to be saved
def save_plots(history: Dict, predictions: Dict, output_dir: Path):
    """
    Save training plots.
    
    Args:
        history: Training history
        predictions: Prediction results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
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

def get_data():
    # TODO: Implement data retrieval
    pass

def prepare_data(df):
    # TODO: Implement data parsing
    pass

def main():
    """Main training function."""
    print("Starting Air Quality Prediction Training...")
    
    # Create directories
    base_dir = Path(__file__).parent.parent.parent
    models_dir = base_dir / "models"
    outputs_dir = base_dir / "outputs"
    models_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    
    # Prepare data
    print("Getting data...")
    df = get_data(n_samples=1000)
    train_loader, test_loader = prepare_data(df)
    
    model = None  # TODO: Initialize your model here

    # Train model
    print("Training model...")
    history = train_model(
        model, train_loader, test_loader
    )
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"air_quality_model_{timestamp}.pt"
    print(f"Model saved to {model_path}")
    # TODO: Save to Hopsworks
    
    # Generate predictions
    print("Generating predictions...")
    predictions = generate_predictions(model, test_loader)
    
    # Save plots
    print("Saving plots...")
    save_plots(history, predictions, outputs_dir)
    
    # Save results metadata
    results = {
        'timestamp': timestamp,
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
    
    return results


if __name__ == "__main__":
    main()
