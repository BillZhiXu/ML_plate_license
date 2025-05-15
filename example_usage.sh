#!/bin/bash

# Example 1: Process a single image
echo "Example 1: Process a single image"
echo "python inference.py --model_path best_model_with_slicedropout.pth --single_image path/to/your/license_plate.jpg"
echo ""

# Example 2: Process a single image with custom output filename
echo "Example 2: Process a single image with custom output filename"
echo "python inference.py --model_path best_model_with_slicedropout.pth --single_image path/to/your/license_plate.jpg --output my_result.png"
echo ""

# Example 3: Process validation samples
echo "Example 3: Process validation samples"
echo "python inference.py --model_path best_model_with_slicedropout.pth"
echo ""

# Example 4: Process more validation samples
echo "Example 4: Process more validation samples"
echo "python inference.py --model_path best_model_with_slicedropout.pth --num_samples 10"
echo ""

# Example 5: Use mock inference if no model is available
echo "Example 5: Use mock inference if no model is available"
echo "python inference_demo.py --mock"
echo ""

# Example 6: Process a single image with mock inference
echo "Example 6: Process a single image with mock inference"
echo "python inference_demo.py --mock --single_image path/to/your/license_plate.jpg"
echo ""

echo "Note: Replace 'path/to/your/license_plate.jpg' with the actual path to your image file."
echo "      Replace 'best_model_with_slicedropout.pth' with your actual model file if different." 