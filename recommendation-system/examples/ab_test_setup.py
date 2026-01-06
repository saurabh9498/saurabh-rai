#!/usr/bin/env python3
"""
A/B Test Setup Example

This example demonstrates how to:
1. Configure an A/B test between recommendation models
2. Set up traffic allocation
3. Track experiment metrics
4. Analyze results

Usage:
    python examples/ab_test_setup.py --create --name "two_tower_v2"
    python examples/ab_test_setup.py --status --experiment-id exp_12345
    python examples/ab_test_setup.py --analyze --experiment-id exp_12345
"""

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.serving.ab_testing import ABTestManager, Experiment, Variant
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_experiment(
    name: str,
    description: str,
    control_model: str,
    treatment_model: str,
    traffic_split: float = 0.5,
    config_path: str = "configs/serving_config.yaml"
) -> Experiment:
    """
    Create a new A/B test experiment.
    
    Args:
        name: Experiment name
        description: Experiment description
        control_model: Path to control model
        treatment_model: Path to treatment model
        traffic_split: Fraction of traffic for treatment (0.0-1.0)
        config_path: Path to configuration
        
    Returns:
        Created experiment
    """
    config = load_config(config_path)
    ab_manager = ABTestManager(config)
    
    # Define variants
    control = Variant(
        name="control",
        model_path=control_model,
        traffic_weight=1.0 - traffic_split,
        description="Current production model"
    )
    
    treatment = Variant(
        name="treatment",
        model_path=treatment_model,
        traffic_weight=traffic_split,
        description="New candidate model"
    )
    
    # Create experiment
    experiment = Experiment(
        id=f"exp_{uuid.uuid4().hex[:8]}",
        name=name,
        description=description,
        variants=[control, treatment],
        metrics=[
            "click_through_rate",
            "conversion_rate",
            "revenue_per_user",
            "diversity_score",
            "latency_p50",
            "latency_p99"
        ],
        start_date=datetime.utcnow(),
        min_sample_size=10000,
        confidence_level=0.95
    )
    
    # Register experiment
    ab_manager.create_experiment(experiment)
    
    logger.info(f"Created experiment: {experiment.id}")
    logger.info(f"  Name: {experiment.name}")
    logger.info(f"  Control: {control_model} ({(1-traffic_split)*100:.0f}%)")
    logger.info(f"  Treatment: {treatment_model} ({traffic_split*100:.0f}%)")
    
    return experiment


def get_experiment_status(
    experiment_id: str,
    config_path: str = "configs/serving_config.yaml"
) -> Dict[str, Any]:
    """
    Get current status of an experiment.
    
    Args:
        experiment_id: Experiment ID
        config_path: Path to configuration
        
    Returns:
        Experiment status dictionary
    """
    config = load_config(config_path)
    ab_manager = ABTestManager(config)
    
    experiment = ab_manager.get_experiment(experiment_id)
    
    if not experiment:
        raise ValueError(f"Experiment not found: {experiment_id}")
    
    status = {
        "id": experiment.id,
        "name": experiment.name,
        "status": experiment.status,
        "start_date": experiment.start_date.isoformat(),
        "variants": [],
        "current_metrics": {}
    }
    
    for variant in experiment.variants:
        variant_stats = ab_manager.get_variant_stats(experiment_id, variant.name)
        status["variants"].append({
            "name": variant.name,
            "traffic_weight": variant.traffic_weight,
            "sample_size": variant_stats.get("sample_size", 0),
            "metrics": variant_stats.get("metrics", {})
        })
    
    # Calculate statistical significance
    significance = ab_manager.calculate_significance(experiment_id)
    status["statistical_significance"] = significance
    
    return status


def analyze_experiment(
    experiment_id: str,
    config_path: str = "configs/serving_config.yaml"
) -> Dict[str, Any]:
    """
    Analyze experiment results and provide recommendations.
    
    Args:
        experiment_id: Experiment ID
        config_path: Path to configuration
        
    Returns:
        Analysis results with recommendations
    """
    config = load_config(config_path)
    ab_manager = ABTestManager(config)
    
    # Get full analysis
    analysis = ab_manager.analyze_experiment(experiment_id)
    
    # Generate recommendation
    recommendation = "CONTINUE"  # Default
    
    if analysis["sufficient_sample_size"]:
        if analysis["statistical_significance"]["is_significant"]:
            if analysis["treatment_lift"]["primary_metric"] > 0:
                recommendation = "PROMOTE_TREATMENT"
            else:
                recommendation = "KEEP_CONTROL"
        else:
            recommendation = "CONTINUE"
    
    analysis["recommendation"] = recommendation
    analysis["recommendation_reason"] = get_recommendation_reason(analysis)
    
    return analysis


def get_recommendation_reason(analysis: Dict) -> str:
    """Generate human-readable recommendation reason."""
    
    if not analysis["sufficient_sample_size"]:
        return (f"Insufficient sample size. Current: {analysis['current_sample_size']}, "
                f"Required: {analysis['min_sample_size']}")
    
    if not analysis["statistical_significance"]["is_significant"]:
        p_value = analysis["statistical_significance"]["p_value"]
        return f"Results not statistically significant (p-value: {p_value:.4f})"
    
    lift = analysis["treatment_lift"]["primary_metric"]
    if lift > 0:
        return f"Treatment shows {lift*100:.2f}% improvement with statistical significance"
    else:
        return f"Treatment shows {abs(lift)*100:.2f}% decline with statistical significance"


def print_experiment_report(analysis: Dict):
    """Print formatted experiment report."""
    
    print("\n" + "="*60)
    print(f"A/B TEST ANALYSIS REPORT")
    print("="*60)
    
    print(f"\nExperiment: {analysis['experiment_name']}")
    print(f"ID: {analysis['experiment_id']}")
    print(f"Duration: {analysis['duration_days']} days")
    
    print(f"\n{'Variant':<15} {'Samples':>10} {'CTR':>10} {'Conv Rate':>12} {'Revenue':>10}")
    print("-"*60)
    
    for variant in analysis['variants']:
        print(f"{variant['name']:<15} "
              f"{variant['sample_size']:>10,} "
              f"{variant['metrics'].get('ctr', 0)*100:>9.2f}% "
              f"{variant['metrics'].get('conversion_rate', 0)*100:>11.2f}% "
              f"${variant['metrics'].get('revenue_per_user', 0):>9.2f}")
    
    print(f"\n{'Metric':<20} {'Lift':>10} {'P-Value':>10} {'Significant':>12}")
    print("-"*55)
    
    for metric, data in analysis['treatment_lift'].items():
        if isinstance(data, dict):
            sig = "✅ Yes" if data.get('is_significant', False) else "❌ No"
            print(f"{metric:<20} {data['lift']*100:>+9.2f}% {data['p_value']:>10.4f} {sig:>12}")
    
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: {analysis['recommendation']}")
    print(f"Reason: {analysis['recommendation_reason']}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="A/B Test Management"
    )
    
    # Action options
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--create",
        action="store_true",
        help="Create a new experiment"
    )
    action_group.add_argument(
        "--status",
        action="store_true",
        help="Get experiment status"
    )
    action_group.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze experiment results"
    )
    action_group.add_argument(
        "--stop",
        action="store_true",
        help="Stop an experiment"
    )
    
    # Experiment parameters
    parser.add_argument(
        "--experiment-id",
        type=str,
        help="Experiment ID (for status/analyze/stop)"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Experiment name (for create)"
    )
    parser.add_argument(
        "--description",
        type=str,
        default="A/B test experiment",
        help="Experiment description"
    )
    parser.add_argument(
        "--control-model",
        type=str,
        default="models/checkpoints/two_tower/v1.pt",
        help="Control model path"
    )
    parser.add_argument(
        "--treatment-model",
        type=str,
        default="models/checkpoints/two_tower/v2.pt",
        help="Treatment model path"
    )
    parser.add_argument(
        "--traffic-split",
        type=float,
        default=0.5,
        help="Traffic fraction for treatment (default: 0.5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    if args.create:
        if not args.name:
            parser.error("--name is required for --create")
        
        experiment = create_experiment(
            name=args.name,
            description=args.description,
            control_model=args.control_model,
            treatment_model=args.treatment_model,
            traffic_split=args.traffic_split
        )
        
        print(f"\n✅ Experiment created successfully!")
        print(f"   ID: {experiment.id}")
        print(f"   Use --experiment-id {experiment.id} for status/analyze")
        
    elif args.status:
        if not args.experiment_id:
            parser.error("--experiment-id is required for --status")
        
        status = get_experiment_status(args.experiment_id)
        
        print(json.dumps(status, indent=2, default=str))
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(status, f, indent=2, default=str)
        
    elif args.analyze:
        if not args.experiment_id:
            parser.error("--experiment-id is required for --analyze")
        
        analysis = analyze_experiment(args.experiment_id)
        
        print_experiment_report(analysis)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"Full analysis saved to {args.output}")
    
    elif args.stop:
        if not args.experiment_id:
            parser.error("--experiment-id is required for --stop")
        
        config = load_config("configs/serving_config.yaml")
        ab_manager = ABTestManager(config)
        ab_manager.stop_experiment(args.experiment_id)
        
        print(f"✅ Experiment {args.experiment_id} stopped")


if __name__ == "__main__":
    main()
