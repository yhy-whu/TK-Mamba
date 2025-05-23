import os
import clip
import torch


## PAOT
ORGAN_NAME = ['Liver','Pancreas','Hepatic Vessel','Kidney','Kidney Cyst','Liver Tumor','Lung Tumor','Pancreas Tumor','Hepatic Vessel Tumor','Colon Tumor','Kidney Tumor']
# Load the model

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


text_inputs = torch.cat([clip.tokenize(f'A computerized tomography of a {item}') for item in ORGAN_NAME]).to(device)

# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(text_features.shape, text_features.dtype)
    torch.save(text_features, '/root/autodl-tmp/TK_Mamba/pretrained_weights/txt_encoding.pth')

