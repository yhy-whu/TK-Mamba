import os
import torch
from transformers import CLIPProcessor, CLIPModel  # Import Hugging Face library

## PAOT
ORGAN_NAME = ['Liver','Pancreas','Hepatic Vessel','Kidney','Kidney Cyst','Liver Tumor','Lung Tumor','Pancreas Tumor','Hepatic Vessel Tumor','Colon Tumor','Kidney Tumor']

# 1. Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load Med-CLIP model and processor
# --- This is the modified line ---
model_id = "flaviagiammarino/pubmed-clip-vit-base-patch32"
# --- End of modification ---

print(f"Loading model from '{model_id}', please wait...")

try:
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model! Please ensure your 'transformers' library is up to date (pip install -U transformers).")
    print(f"Also, check your network connection to the Hugging Face hub.")
    print(f"Error details: {e}")
    exit()

# 3. Create a list of text prompts
text_prompts = [f'A computed tomography of a {item}' for item in ORGAN_NAME]

# 4. Process the text using Med-CLIP's processor
inputs = processor(
    text=text_prompts, 
    return_tensors="pt", 
    padding=True, 
    truncation=True
).to(device)

# 5. Calculate text embedding features
with torch.no_grad():
    text_features = model.get_text_features(**inputs)
    
    print(f"Feature shape: {text_features.shape}, Data type: {text_features.dtype}")
    
    # Normalization (Optional, but
    #  CLIP often does this)
    # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    save_path = '/root/autodl-tmp/TK_Mamba/pretrained_weights/txt_encoding.pth'
    torch.save(text_features, save_path)

print(f"Med-CLIP text features successfully saved to: {save_path}")