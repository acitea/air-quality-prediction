# Air Quality Prediction 🌍

A PyTorch-based machine learning project for predicting PM2.5 air quality levels using environmental and traffic data. The project features automated daily training via GitHub Actions and real-time results visualization through GitHub Pages.

## Features

- 🤖 **PyTorch Neural Network**: Deep learning model for PM2.5 prediction
- 📦 **UV Package Manager**: Modern, fast Python package management
- ⏰ **Automated Training**: Daily scheduled training via GitHub Actions
- 📊 **Interactive Dashboard**: Real-time results on GitHub Pages
- 📈 **Performance Tracking**: Visualizations of model performance and predictions

## Model Architecture

The air quality prediction model uses a feedforward neural network with:
- **Input Layer**: 5 features (temperature, humidity, wind speed, pressure, traffic density)
- **Hidden Layers**: 2 layers with 64 neurons each, ReLU activation, and dropout
- **Output Layer**: Single output (PM2.5 concentration in µg/m³)

## Installation

### Prerequisites

- Python 3.12+
- [UV package manager](https://github.com/astral-sh/uv)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/acitea/air-quality-prediction.git
cd air-quality-prediction
```

2. Install UV (if not already installed):
```bash
pip install uv
```

3. Install dependencies:
```bash
uv sync
```

## Usage

### Training the Model

Run the training pipeline:
```bash
uv run python -m air_quality_prediction.train
```

Or use the package entry point:
```bash
uv run air-quality-prediction
```

### View Results

After training, check the generated files:
- `models/air_quality_model_*.pt` - Trained model weights
- `outputs/latest_results.json` - Training metrics and predictions
- `outputs/training_history.png` - Training/validation loss plot
- `outputs/predictions.png` - Predicted vs actual values plot

### GitHub Pages Dashboard

Visit the GitHub Pages site to see interactive visualizations of the latest predictions and model performance: [View Dashboard](https://acitea.github.io/air-quality-prediction/)

## Project Structure

```
air-quality-prediction/
├── .github/
│   └── workflows/
│       ├── train.yml           # Daily training workflow
│       └── deploy-pages.yml    # GitHub Pages deployment
├── src/
│   └── air_quality_prediction/
│       ├── __init__.py         # Package entry point
│       ├── model.py            # PyTorch model definition
│       ├── data_utils.py       # Data generation and processing
│       └── train.py            # Training pipeline
├── data/                       # Data directory
├── models/                     # Saved model checkpoints
├── outputs/                    # Training outputs and visualizations
├── docs/                       # GitHub Pages site
│   ├── index.html             # Dashboard HTML
│   └── outputs/               # Copied outputs for web display
├── pyproject.toml             # UV project configuration
└── README.md                  # This file
```

## Automated Training

The project uses GitHub Actions to automatically train the model daily at 2:00 AM UTC. The workflow:
1. Checks out the latest code
2. Sets up Python and UV
3. Installs dependencies
4. Runs the training pipeline
5. Uploads model artifacts
6. Commits and pushes results to the repository

You can also manually trigger training from the Actions tab in GitHub.

## Development

### Running Tests

Currently, the project focuses on the core training pipeline. Tests can be added in future iterations.

### Adding New Features

The modular structure makes it easy to:
- Modify the model architecture in `model.py`
- Add new features in `data_utils.py`
- Customize training parameters in `train.py`
- Update the dashboard in `docs/index.html`

## Technologies

- **PyTorch**: Deep learning framework
- **UV**: Python package manager
- **NumPy & Pandas**: Data manipulation
- **Scikit-learn**: Utilities and metrics
- **Matplotlib**: Visualization
- **GitHub Actions**: CI/CD automation
- **GitHub Pages**: Web hosting

## License

This project is available for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Built with modern Python tooling and best practices for reproducible machine learning.