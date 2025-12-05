import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from PIL import Image
from pathlib import Path
import mobileclip
import time

# Load model with checkpoint from local checkpoints folder
checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "mobileclip_s0.pt"
model, _, preprocess = mobileclip.create_model_and_transforms(
    "mobileclip_s0", 
    pretrained=str(checkpoint_path)
)
tokenizer = mobileclip.get_tokenizer("mobileclip_s0")

# Get all images from img folder
img_folder = Path(__file__).parent.parent / "img"
image_files = list(img_folder.glob("*.png")) + list(img_folder.glob("*.jpg")) + list(img_folder.glob("*.jpeg"))

if not image_files:
    print("No images found in img folder!")
    exit(1)

# Define text labels
text_labels = ["ramp","stair case", "door","grated floor"]
text = tokenizer(text_labels)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print(f"Processing {len(image_files)} images...\n")

inference_times = []

# Process each image
for img_path in image_files:
    print(f"Processing: {img_path.name}")
    
    # Load and preprocess image
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
    
    # Start timing
    start_time = time.perf_counter()
    
    # Perform inference (only use autocast if CUDA is available)
    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast('cuda'):
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
        else:
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # End timing
    inference_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
    inference_times.append(inference_time)
    
    # Print results
    print(f"Inference time: {inference_time:.2f} ms")
    print("Label probabilities:")
    for label, prob in zip(text_labels, text_probs[0]):
        print(f"  {label}: {prob.item():.2%}")
    print()

# Print summary statistics
if inference_times:
    print("=" * 50)
    print("Inference Time Summary:")
    print(f"  Average: {sum(inference_times) / len(inference_times):.2f} ms")
    print(f"  Min: {min(inference_times):.2f} ms")
    print(f"  Max: {max(inference_times):.2f} ms")
    print("=" * 50)
