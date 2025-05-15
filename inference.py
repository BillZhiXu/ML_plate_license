import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
import random

# Import the model architecture from swin_arg.py
from swin_arg import ImprovedMultiHeadPlateCNN, SliceDropoutModule

def load_model(model_path, device, use_slice_dropout=True):
    """
    Load the trained model from the specified path
    """
    # Define model parameters (must match the trained model)
    img_height = 64
    img_width = 192
    max_length = 7
    
    # Create model instance
    model = ImprovedMultiHeadPlateCNN(
        in_channels=3,
        num_classes=37,  # 0-9, A-Z, and padding
        num_heads=max_length,
        input_size=(img_height, img_width),
        use_slice_dropout=use_slice_dropout
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

def get_transform(img_height=64, img_width=192):
    """
    Get the validation transform for inference
    """
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    return transform

def load_validation_samples(csv_path, img_dir, num_samples=5, seed=42):
    """
    Load a random selection of samples from the validation set
    """
    import pandas as pd
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read validation CSV
    df = pd.read_csv(csv_path)
    
    # Randomly select samples
    if len(df) <= num_samples:
        selected_samples = df
    else:
        selected_samples = df.sample(num_samples, random_state=seed)
    
    samples = []
    for _, row in selected_samples.iterrows():
        img_path = os.path.join(img_dir, row['image_path'])
        true_label = row['plate_text']
        
        try:
            image = Image.open(img_path).convert('RGB')
            samples.append((image, true_label, img_path))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    return samples

def load_single_image(image_path):
    """
    Load a single image for inference
    """
    try:
        image = Image.open(image_path).convert('RGB')
        # For a single image, we don't have a ground truth label
        # Use the filename without extension as a placeholder
        filename = os.path.basename(image_path)
        return [(image, "Unknown", image_path)]
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return []

def run_inference(model, samples, transform, device):
    """
    Run inference on the samples
    """
    results = []
    char_map = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
    
    for image, true_label, img_path in samples:
        # Apply transform
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            logits = model(img_tensor)
            
            # Get predictions
            pred_label = ""
            for head_idx, output in enumerate(logits):
                _, pred_idx = torch.max(output[0], dim=0)
                char = char_map[pred_idx.item()]
                
                # Only add non-padding characters
                if char != '_':
                    pred_label += char
        
        # Store results
        results.append({
            'image': image,
            'true_label': true_label,
            'pred_label': pred_label,
            'img_path': img_path
        })
    
    return results

def visualize_results(results):
    """
    Visualize the inference results
    """
    num_samples = len(results)
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for i, result in enumerate(axes):
        # Get the sample
        sample = results[i]
        
        # Display the image
        result.imshow(np.array(sample['image']))
        
        # Add title with true and predicted labels
        if sample['true_label'] == "Unknown":
            # For single image mode, only show the prediction
            result.set_title(f"Predicted License Plate: {sample['pred_label']}")
        else:
            # For validation samples, show both true and predicted
            match_status = "✓" if sample['true_label'] == sample['pred_label'] else "✗"
            result.set_title(f"True: {sample['true_label']} | Pred: {sample['pred_label']} {match_status}")
        
        result.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='License Plate Recognition Inference')
    parser.add_argument('--model_path', type=str, default='best_model_with_slicedropout.pth',
                        help='Path to the trained model weights')
    parser.add_argument('--val_csv', type=str, default='data/arg_plate_dataset/valid_anotaciones.csv',
                        help='Path to validation CSV file')
    parser.add_argument('--img_dir', type=str, default='data/arg_plate_dataset',
                        help='Directory containing the images')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of validation samples to visualize')
    parser.add_argument('--output', type=str, default='inference_results.png',
                        help='Output image filename')
    parser.add_argument('--no_slice_dropout', action='store_true',
                        help='Disable slice dropout for inference')
    parser.add_argument('--single_image', type=str, default=None,
                        help='Path to a single image file for license plate recognition')
    args = parser.parse_args()
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    use_slice_dropout = not args.no_slice_dropout
    model = load_model(args.model_path, device, use_slice_dropout)
    print(f"Model loaded from {args.model_path}")
    
    # Get transform
    transform = get_transform()
    
    # Determine if we're processing a single image or validation samples
    if args.single_image:
        # Single image mode
        print(f"Processing single image: {args.single_image}")
        samples = load_single_image(args.single_image)
        if not samples:
            print("Failed to load the image. Exiting.")
            return
        
        # Use the image filename for the output
        if args.output == 'inference_results.png':
            base_name = os.path.splitext(os.path.basename(args.single_image))[0]
            args.output = f"{base_name}_result.png"
    else:
        # Validation samples mode
        samples = load_validation_samples(args.val_csv, args.img_dir, args.num_samples)
        print(f"Loaded {len(samples)} validation samples")
    
    if not samples:
        print("No samples to process. Exiting.")
        return
    
    # Run inference
    results = run_inference(model, samples, transform, device)
    
    # Visualize results
    fig = visualize_results(results)
    
    # Save the visualization
    plt.savefig(args.output)
    print(f"Results saved to {args.output}")
    
    # Print summary
    if args.single_image:
        print(f"\nPredicted license plate: {results[0]['pred_label']}")
    else:
        correct = sum(1 for r in results if r['true_label'] == r['pred_label'])
        print(f"Accuracy: {correct}/{len(results)} ({correct/len(results)*100:.2f}%)")
        
        # Print detailed results
        print("\nDetailed Results:")
        for i, result in enumerate(results):
            match_status = "✓" if result['true_label'] == result['pred_label'] else "✗"
            print(f"{i+1}. File: {os.path.basename(result['img_path'])}")
            print(f"   True: {result['true_label']} | Pred: {result['pred_label']} {match_status}")
    
    plt.show()

if __name__ == "__main__":
    main() 