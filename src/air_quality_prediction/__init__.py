"""Air Quality Prediction package."""

from .train import main as train_main


def main() -> None:
    """Entry point for the air quality prediction training."""
    train_main()
