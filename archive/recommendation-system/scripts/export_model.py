#!/usr/bin/env python3
"""
Model Export Script

Exports trained models to production formats:
- ONNX for cross-platform deployment
- TorchScript for PyTorch serving
- Triton model repository format

Usage:
    python scripts/export_model.py --model two_tower --checkpoint ./models/two_tower.pt --format onnx
    python scripts/export_model.py --model dlrm --checkpoint ./models/dlrm.pt --format triton
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import shutil

import torch
import torch.onnx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.two_tower import TwoTowerModel, TwoTowerConfig
from src.models.dlrm import DLRM, DLRMConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelExporter:
    """Handles model export to various formats."""
    
    SUPPORTED_FORMATS = ['onnx', 'torchscript', 'triton']
    SUPPORTED_MODELS = ['two_tower', 'dlrm', 'two_tower_user', 'two_tower_item']
    
    def __init__(
        self,
        model_type: str,
        checkpoint_path: str,
        output_dir: str,
        config_path: Optional[str] = None,
    ):
        self.model_type = model_type
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.config_path = Path(config_path) if config_path else None
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.config = None
        
    def load_model(self) -> torch.nn.Module:
        """Load model from checkpoint."""
        logger.info(f"Loading {self.model_type} model from {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Load config
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
        elif self.config_path and self.config_path.exists():
            import yaml
            with open(self.config_path) as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError("No config found in checkpoint or config path")
        
        # Create model based on type
        if self.model_type in ['two_tower', 'two_tower_user', 'two_tower_item']:
            self.config = TwoTowerConfig(**config_dict)
            self.model = TwoTowerModel(self.config)
        elif self.model_type == 'dlrm':
            self.config = DLRMConfig(**config_dict)
            self.model = DLRM(self.config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logger.info(f"Model loaded successfully")
        
        return self.model
    
    def get_sample_inputs(self) -> Dict[str, torch.Tensor]:
        """Generate sample inputs for tracing."""
        batch_size = 1
        
        if self.model_type == 'two_tower':
            return {
                'user_categorical': torch.randint(0, 100, (batch_size, self.config.num_user_categorical_features)),
                'user_dense': torch.randn(batch_size, self.config.num_user_dense_features),
                'user_history': torch.randint(0, 1000, (batch_size, self.config.max_history_length)),
                'history_mask': torch.ones(batch_size, self.config.max_history_length),
                'item_categorical': torch.randint(0, 100, (batch_size, self.config.num_item_categorical_features)),
                'item_dense': torch.randn(batch_size, self.config.num_item_dense_features),
            }
        elif self.model_type == 'two_tower_user':
            return {
                'user_categorical': torch.randint(0, 100, (batch_size, self.config.num_user_categorical_features)),
                'user_dense': torch.randn(batch_size, self.config.num_user_dense_features),
                'user_history': torch.randint(0, 1000, (batch_size, self.config.max_history_length)),
                'history_mask': torch.ones(batch_size, self.config.max_history_length),
            }
        elif self.model_type == 'two_tower_item':
            return {
                'item_categorical': torch.randint(0, 100, (batch_size, self.config.num_item_categorical_features)),
                'item_dense': torch.randn(batch_size, self.config.num_item_dense_features),
            }
        elif self.model_type == 'dlrm':
            return {
                'sparse_features': torch.randint(0, 100, (batch_size, self.config.num_sparse_features)),
                'dense_features': torch.randn(batch_size, self.config.num_dense_features),
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def export_onnx(
        self,
        opset_version: int = 14,
        dynamic_axes: bool = True,
    ) -> Path:
        """Export model to ONNX format."""
        logger.info("Exporting to ONNX format...")
        
        output_path = self.output_dir / f"{self.model_type}.onnx"
        sample_inputs = self.get_sample_inputs()
        
        # Prepare input names and dynamic axes
        input_names = list(sample_inputs.keys())
        output_names = ['output']
        
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {name: {0: 'batch_size'} for name in input_names}
            dynamic_axes_dict['output'] = {0: 'batch_size'}
        
        # Export
        torch.onnx.export(
            self.model,
            tuple(sample_inputs.values()),
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_dict,
            opset_version=opset_version,
            do_constant_folding=True,
        )
        
        logger.info(f"ONNX model saved to {output_path}")
        
        # Verify ONNX model
        self._verify_onnx(output_path, sample_inputs)
        
        return output_path
    
    def _verify_onnx(self, onnx_path: Path, sample_inputs: Dict[str, torch.Tensor]):
        """Verify ONNX model produces correct outputs."""
        try:
            import onnx
            import onnxruntime as ort
            
            # Check model validity
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation passed")
            
            # Compare outputs
            session = ort.InferenceSession(str(onnx_path))
            
            onnx_inputs = {k: v.numpy() for k, v in sample_inputs.items()}
            onnx_outputs = session.run(None, onnx_inputs)
            
            with torch.no_grad():
                torch_outputs = self.model(**sample_inputs)
                if isinstance(torch_outputs, tuple):
                    torch_outputs = torch_outputs[0]
            
            np.testing.assert_allclose(
                torch_outputs.numpy(),
                onnx_outputs[0],
                rtol=1e-3,
                atol=1e-5,
            )
            logger.info("ONNX output verification passed")
            
        except ImportError:
            logger.warning("onnx/onnxruntime not installed, skipping verification")
    
    def export_torchscript(self, method: str = 'trace') -> Path:
        """Export model to TorchScript format."""
        logger.info(f"Exporting to TorchScript ({method})...")
        
        output_path = self.output_dir / f"{self.model_type}.pt"
        sample_inputs = self.get_sample_inputs()
        
        if method == 'trace':
            scripted = torch.jit.trace(
                self.model,
                example_inputs=tuple(sample_inputs.values()),
            )
        elif method == 'script':
            scripted = torch.jit.script(self.model)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        scripted.save(str(output_path))
        logger.info(f"TorchScript model saved to {output_path}")
        
        # Verify
        self._verify_torchscript(output_path, sample_inputs)
        
        return output_path
    
    def _verify_torchscript(self, ts_path: Path, sample_inputs: Dict[str, torch.Tensor]):
        """Verify TorchScript model."""
        loaded = torch.jit.load(str(ts_path))
        
        with torch.no_grad():
            original_output = self.model(**sample_inputs)
            loaded_output = loaded(**sample_inputs)
            
            if isinstance(original_output, tuple):
                original_output = original_output[0]
                loaded_output = loaded_output[0]
            
            torch.testing.assert_close(original_output, loaded_output)
        
        logger.info("TorchScript verification passed")
    
    def export_triton(
        self,
        model_name: Optional[str] = None,
        max_batch_size: int = 64,
        platform: str = 'onnxruntime_onnx',
    ) -> Path:
        """Export model to Triton Inference Server format."""
        logger.info("Exporting to Triton model repository format...")
        
        model_name = model_name or self.model_type
        model_dir = self.output_dir / model_name
        version_dir = model_dir / "1"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Export ONNX model
        onnx_path = self.export_onnx()
        shutil.copy(onnx_path, version_dir / "model.onnx")
        
        # Create config.pbtxt
        config = self._generate_triton_config(model_name, max_batch_size, platform)
        config_path = model_dir / "config.pbtxt"
        
        with open(config_path, 'w') as f:
            f.write(config)
        
        logger.info(f"Triton model repository created at {model_dir}")
        
        return model_dir
    
    def _generate_triton_config(
        self,
        model_name: str,
        max_batch_size: int,
        platform: str,
    ) -> str:
        """Generate Triton config.pbtxt."""
        sample_inputs = self.get_sample_inputs()
        
        # Build input configs
        input_configs = []
        for name, tensor in sample_inputs.items():
            dims = list(tensor.shape[1:])  # Exclude batch dim
            dtype = "TYPE_INT64" if tensor.dtype == torch.long else "TYPE_FP32"
            input_configs.append(f'''
input {{
    name: "{name}"
    data_type: {dtype}
    dims: {dims}
}}''')
        
        # Build config
        config = f'''name: "{model_name}"
platform: "{platform}"
max_batch_size: {max_batch_size}

{"".join(input_configs)}

output {{
    name: "output"
    data_type: TYPE_FP32
    dims: [-1]
}}

instance_group {{
    count: 1
    kind: KIND_GPU
}}

dynamic_batching {{
    preferred_batch_size: [16, 32, 64]
    max_queue_delay_microseconds: 100
}}

optimization {{
    execution_accelerators {{
        gpu_execution_accelerator: [
            {{ name: "tensorrt" }}
        ]
    }}
}}
'''
        return config


def main():
    parser = argparse.ArgumentParser(description='Export recommendation models')
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=ModelExporter.SUPPORTED_MODELS,
        help='Model type to export',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./exported_models',
        help='Output directory',
    )
    parser.add_argument(
        '--format',
        type=str,
        default='onnx',
        choices=ModelExporter.SUPPORTED_FORMATS,
        help='Export format',
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to model config (if not in checkpoint)',
    )
    parser.add_argument(
        '--model-name',
        type=str,
        help='Model name for Triton export',
    )
    parser.add_argument(
        '--max-batch-size',
        type=int,
        default=64,
        help='Max batch size for Triton',
    )
    
    args = parser.parse_args()
    
    exporter = ModelExporter(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
    )
    
    exporter.load_model()
    
    if args.format == 'onnx':
        exporter.export_onnx()
    elif args.format == 'torchscript':
        exporter.export_torchscript()
    elif args.format == 'triton':
        exporter.export_triton(
            model_name=args.model_name,
            max_batch_size=args.max_batch_size,
        )
    
    logger.info("Export completed successfully!")


if __name__ == '__main__':
    main()
