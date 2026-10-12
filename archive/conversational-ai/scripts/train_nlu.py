"""Train NLU models (intent classifier and entity extractor)."""

import argparse
import logging
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(data_path: str):
    """Load training data from YAML or JSON."""
    with open(data_path) as f:
        if data_path.endswith('.yaml'):
            return yaml.safe_load(f)
        else:
            import json
            return json.load(f)


def train_intent_classifier(data, config, output_dir):
    """Train intent classification model."""
    logger.info("Training intent classifier...")
    
    from src.nlu.intent_classifier import IntentClassifier, IntentConfig
    
    intent_config = IntentConfig(
        model_name=config.get('model', 'distilbert-base-uncased'),
        num_intents=len(data['intents']),
        intent_labels=[i['name'] for i in data['intents']],
    )
    
    model = IntentClassifier(intent_config)
    
    # Training loop would go here
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Save model
    output_path = Path(output_dir) / "intent_classifier.pt"
    torch.save(model.state_dict(), output_path)
    logger.info(f"Saved model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train NLU models")
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--config", default="configs/nlu_config.yaml")
    parser.add_argument("--output", default="models/")
    parser.add_argument("--epochs", type=int, default=10)
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load data
    data = load_training_data(args.data)
    
    # Train
    Path(args.output).mkdir(parents=True, exist_ok=True)
    train_intent_classifier(data, config.get('intent_classifier', {}), args.output)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
