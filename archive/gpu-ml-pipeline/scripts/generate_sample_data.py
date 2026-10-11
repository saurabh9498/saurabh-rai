#!/usr/bin/env python3
"""
Sample Data Generator for GPU-Accelerated ML Pipeline

Generates synthetic data for development and testing:
- Sample images (synthetic patterns)
- Pre-batched numpy arrays
- Dummy ONNX models
- Calibration data
- Expected output templates

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --force           # Overwrite existing
    python scripts/generate_sample_data.py --calibration-only
    python scripts/generate_sample_data.py --num-images 1000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple, List
import struct

# Optional imports with fallbacks
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not installed. Some features disabled.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: Pillow not installed. Image generation disabled.")


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def create_synthetic_image(
    width: int = 224,
    height: int = 224,
    pattern: str = "gradient"
) -> bytes:
    """
    Create a synthetic test image.
    
    Args:
        width: Image width
        height: Image height
        pattern: Pattern type ('gradient', 'checkerboard', 'noise', 'solid')
    
    Returns:
        Image bytes in PNG format
    """
    if not HAS_NUMPY or not HAS_PIL:
        # Return a minimal valid PNG
        return create_minimal_png(width, height)
    
    if pattern == "gradient":
        # Create RGB gradient
        r = np.tile(np.linspace(0, 255, width), (height, 1))
        g = np.tile(np.linspace(0, 255, height).reshape(-1, 1), (1, width))
        b = np.full((height, width), 128)
        img_array = np.stack([r, g, b], axis=2).astype(np.uint8)
    
    elif pattern == "checkerboard":
        # Create checkerboard pattern
        block_size = 28
        x = np.arange(width) // block_size
        y = np.arange(height) // block_size
        pattern = (x[np.newaxis, :] + y[:, np.newaxis]) % 2
        img_array = (pattern * 255).astype(np.uint8)
        img_array = np.stack([img_array] * 3, axis=2)
    
    elif pattern == "noise":
        # Create random noise
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    else:  # solid
        # Solid color
        img_array = np.full((height, width, 3), 128, dtype=np.uint8)
    
    # Convert to PIL and save to bytes
    img = Image.fromarray(img_array, mode='RGB')
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=90)
    return buffer.getvalue()


def create_minimal_png(width: int, height: int) -> bytes:
    """Create a minimal valid PNG file without dependencies."""
    import zlib
    
    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk_len = struct.pack('>I', len(data))
        chunk_crc = struct.pack('>I', zlib.crc32(chunk_type + data) & 0xffffffff)
        return chunk_len + chunk_type + data + chunk_crc
    
    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'
    
    # IHDR chunk
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    ihdr = png_chunk(b'IHDR', ihdr_data)
    
    # IDAT chunk (compressed image data - solid gray)
    raw_data = b''
    for y in range(height):
        raw_data += b'\x00'  # Filter byte
        raw_data += b'\x80\x80\x80' * width  # RGB gray pixels
    
    compressed = zlib.compress(raw_data, 9)
    idat = png_chunk(b'IDAT', compressed)
    
    # IEND chunk
    iend = png_chunk(b'IEND', b'')
    
    return signature + ihdr + idat + iend


def create_dummy_onnx_model(output_path: Path, model_type: str = "classifier") -> None:
    """
    Create a dummy ONNX model for testing.
    
    This creates a minimal valid ONNX file structure.
    For real testing, download actual models.
    """
    # Minimal ONNX file structure (protobuf format)
    # This is a simplified version - real ONNX files are more complex
    
    onnx_header = b'\\x08\\x07'  # ONNX IR version
    
    # Create a placeholder file with metadata
    model_info = {
        "format": "ONNX",
        "version": "1.14.0",
        "model_type": model_type,
        "input_shape": [1, 3, 224, 224],
        "output_shape": [1, 1000] if model_type == "classifier" else [1, 100, 6],
        "note": "This is a placeholder. Download real models for actual inference.",
        "download_instructions": {
            "resnet50": "wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx",
            "yolov8": "pip install ultralytics && python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')\""
        }
    }
    
    # Write as JSON with .onnx.json extension to indicate it's a placeholder
    json_path = output_path.with_suffix('.onnx.json')
    with open(json_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"  ✓ Created model placeholder: {json_path}")
    print(f"    (Download real model using instructions in the JSON file)")


def create_sample_batch(
    output_path: Path,
    batch_size: int = 32,
    channels: int = 3,
    height: int = 224,
    width: int = 224
) -> None:
    """Create a sample batch of preprocessed images as numpy array."""
    if not HAS_NUMPY:
        print(f"  ⚠ Skipping batch creation (numpy not installed)")
        return
    
    # Create random normalized data (simulating preprocessed images)
    batch = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    
    # Normalize to typical ImageNet range
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    batch = (batch * std + mean).astype(np.float32)
    
    np.save(output_path, batch)
    print(f"  ✓ Created batch: {output_path} (shape: {batch.shape})")


def create_expected_outputs(output_path: Path) -> None:
    """Create expected output templates for validation."""
    outputs = {
        "resnet50": {
            "model": "resnet50",
            "input_shape": [1, 3, 224, 224],
            "output_shape": [1, 1000],
            "sample_outputs": {
                "cat.jpg": {
                    "top5_classes": [281, 282, 285, 287, 283],
                    "top5_labels": ["tabby", "tiger_cat", "Egyptian_cat", "lynx", "Persian_cat"],
                    "top5_probs": [0.45, 0.25, 0.12, 0.08, 0.05]
                },
                "dog.jpg": {
                    "top5_classes": [207, 208, 209, 210, 211],
                    "top5_labels": ["golden_retriever", "Labrador", "Chesapeake", "curly-coated", "flat-coated"],
                    "top5_probs": [0.52, 0.18, 0.11, 0.09, 0.06]
                }
            },
            "performance_baseline": {
                "fps_gpu_fp32": 450,
                "fps_gpu_fp16": 890,
                "fps_gpu_int8": 1250,
                "latency_p99_ms": 2.5
            }
        },
        "yolov8n": {
            "model": "yolov8n",
            "input_shape": [1, 3, 640, 640],
            "output_shape": [1, 84, 8400],
            "sample_outputs": {
                "street.jpg": {
                    "num_detections": 12,
                    "classes_detected": ["car", "person", "bicycle", "traffic_light"],
                    "avg_confidence": 0.72
                }
            },
            "performance_baseline": {
                "fps_gpu_fp32": 320,
                "fps_gpu_fp16": 580,
                "fps_gpu_int8": 890,
                "latency_p99_ms": 3.2
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(outputs, f, indent=2)
    
    print(f"  ✓ Created expected outputs: {output_path}")


def create_benchmark_config(output_path: Path) -> None:
    """Create benchmark configuration file."""
    config = {
        "benchmark_settings": {
            "warmup_iterations": 50,
            "benchmark_iterations": 1000,
            "batch_sizes": [1, 2, 4, 8, 16, 32],
            "precisions": ["fp32", "fp16", "int8"],
            "input_shapes": {
                "classification": [3, 224, 224],
                "detection": [3, 640, 640],
                "segmentation": [3, 512, 512]
            }
        },
        "hardware_profiles": {
            "rtx_4090": {
                "compute_capability": "8.9",
                "memory_gb": 24,
                "tensor_cores": 512
            },
            "a100": {
                "compute_capability": "8.0",
                "memory_gb": 80,
                "tensor_cores": 432
            },
            "t4": {
                "compute_capability": "7.5",
                "memory_gb": 16,
                "tensor_cores": 320
            }
        },
        "optimization_targets": {
            "latency_critical": {
                "max_latency_ms": 10,
                "min_throughput_fps": 100
            },
            "throughput_critical": {
                "max_latency_ms": 50,
                "min_throughput_fps": 500
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  ✓ Created benchmark config: {output_path}")


def generate_calibration_images(
    output_dir: Path,
    num_images: int = 100,
    width: int = 224,
    height: int = 224
) -> None:
    """Generate calibration images for INT8 quantization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patterns = ["gradient", "noise", "checkerboard", "solid"]
    
    for i in range(num_images):
        pattern = patterns[i % len(patterns)]
        img_bytes = create_synthetic_image(width, height, pattern)
        
        img_path = output_dir / f"calib_{i:04d}.jpg"
        with open(img_path, 'wb') as f:
            f.write(img_bytes)
    
    print(f"  ✓ Created {num_images} calibration images in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample data for GPU-Accelerated ML Pipeline"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output directory (default: project data/)"
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Overwrite existing files"
    )
    parser.add_argument(
        "--calibration-only", action="store_true",
        help="Only generate calibration data"
    )
    parser.add_argument(
        "--num-images", type=int, default=100,
        help="Number of calibration images to generate"
    )
    parser.add_argument(
        "--image-size", type=int, default=224,
        help="Image size (width and height)"
    )
    args = parser.parse_args()
    
    # Determine output directory
    if args.output:
        data_dir = args.output
    else:
        data_dir = get_project_root() / "data"
    
    print("=" * 60)
    print("GPU-Accelerated ML Pipeline - Sample Data Generator")
    print("=" * 60)
    print(f"\nOutput directory: {data_dir}")
    print()
    
    # Create directories
    (data_dir / "models").mkdir(parents=True, exist_ok=True)
    (data_dir / "sample" / "images").mkdir(parents=True, exist_ok=True)
    (data_dir / "sample" / "batches").mkdir(parents=True, exist_ok=True)
    (data_dir / "sample" / "expected_outputs").mkdir(parents=True, exist_ok=True)
    (data_dir / "calibration" / "calibration_images").mkdir(parents=True, exist_ok=True)
    
    if args.calibration_only:
        print("\n🔧 Generating Calibration Data Only...")
        print("-" * 40)
        generate_calibration_images(
            data_dir / "calibration" / "calibration_images",
            args.num_images,
            args.image_size,
            args.image_size
        )
        print("\n✅ Calibration data generation complete!")
        return
    
    # Generate sample images
    print("\n🖼️  Generating Sample Images...")
    print("-" * 40)
    
    sample_images = [
        ("cat.jpg", "gradient"),
        ("dog.jpg", "noise"),
        ("car.jpg", "checkerboard"),
        ("street.jpg", "gradient"),
    ]
    
    for filename, pattern in sample_images:
        img_path = data_dir / "sample" / "images" / filename
        img_bytes = create_synthetic_image(args.image_size, args.image_size, pattern)
        with open(img_path, 'wb') as f:
            f.write(img_bytes)
        print(f"  ✓ Created: {img_path}")
    
    # Generate model placeholders
    print("\n🤖 Creating Model Placeholders...")
    print("-" * 40)
    create_dummy_onnx_model(data_dir / "models" / "resnet50", "classifier")
    create_dummy_onnx_model(data_dir / "models" / "yolov8n", "detector")
    
    # Generate batch data
    print("\n📦 Generating Batch Data...")
    print("-" * 40)
    
    for batch_size in [1, 8, 32]:
        batch_path = data_dir / "sample" / "batches" / f"batch_{batch_size}.npy"
        create_sample_batch(batch_path, batch_size, 3, args.image_size, args.image_size)
    
    # Generate expected outputs
    print("\n📋 Creating Expected Outputs...")
    print("-" * 40)
    create_expected_outputs(
        data_dir / "sample" / "expected_outputs" / "model_outputs.json"
    )
    create_benchmark_config(
        data_dir / "sample" / "expected_outputs" / "benchmark_config.json"
    )
    
    # Generate calibration images
    print("\n🔧 Generating Calibration Data...")
    print("-" * 40)
    generate_calibration_images(
        data_dir / "calibration" / "calibration_images",
        min(args.num_images, 100),  # Limit for quick generation
        args.image_size,
        args.image_size
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Sample data generation complete!")
    print("=" * 60)
    print(f"""
Files created:
  Sample Images:
    - cat.jpg, dog.jpg, car.jpg, street.jpg

  Model Placeholders:
    - resnet50.onnx.json (download instructions included)
    - yolov8n.onnx.json (download instructions included)

  Batch Data:
    - batch_1.npy, batch_8.npy, batch_32.npy

  Expected Outputs:
    - model_outputs.json
    - benchmark_config.json

  Calibration Data:
    - {min(args.num_images, 100)} calibration images

Next steps:
  1. Download real ONNX models (see model placeholder JSON files)
  2. Build TensorRT engines: python scripts/build_engine.py
  3. Run benchmarks: python scripts/benchmark.py
  4. For more calibration images: --num-images 1000
""")


if __name__ == "__main__":
    main()
