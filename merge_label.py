import os
import glob
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


MERGE_PAIRS = {
    'liver': ['Liver', 'Liver Tumor'],
    'pancreas': ['Pancreas', 'Pancreas Tumor'],
    'hepaticvessel': ['Hepatic Vessel', 'Hepatic Vessel Tumor'],
    'kits23': ['Kidney', 'Kidney Tumor', 'Kidney Cyst']
}

RESULT_ROOT_PATH = "/root/autodl-tmp/TK_Mamba/pre_result/"

def merge_labels(sample_dir, merge_group, output_name):
    sample_name = os.path.basename(sample_dir.rstrip('/'))  
    print(f"Processing sample: {sample_name}")
    
    label_arrays = []
    for label_name in merge_group:
        label_path = os.path.join(sample_dir, f"{sample_name}_{label_name}.nii.gz")
        print(f"Looking for label file: {label_path}")
        if not os.path.exists(label_path):
            print(f"Warning: Label file {label_path} not found, skipping...")
            return
        label_img = sitk.ReadImage(label_path)
        label_array = sitk.GetArrayFromImage(label_img)
        label_arrays.append(label_array)


    merged_label = np.zeros_like(label_arrays[0], dtype=np.int32)
    for idx, label_array in enumerate(label_arrays):

        mask = label_array > 0
        merged_label[mask] = idx + 1  
        print(f"Label {label_name} has {np.sum(mask)} non-zero voxels")


    if np.sum(merged_label > 0) == 0:
        print(f"Warning: Merged label for {sample_name} is all zeros, skipping...")
        return

    merged_label_img = sitk.GetImageFromArray(merged_label)
    merged_label_img.CopyInformation(label_img) 
    output_path = os.path.join(sample_dir, f"{sample_name}_{output_name}.nii.gz")
    sitk.WriteImage(merged_label_img, output_path)
    print(f"Saved merged label: {output_path}")

def main():
    sample_dirs = [d for d in glob.glob(os.path.join(RESULT_ROOT_PATH, "*/")) if os.path.isdir(d)]
    print(f"Found {len(sample_dirs)} sample directories")
    if len(sample_dirs) == 0:
        print(f"No directories found in {RESULT_ROOT_PATH}. Please check the path.")
        return
    print("Sample directories:", [os.path.basename(d.rstrip('/')) for d in sample_dirs[:5]]) 

    for sample_dir in tqdm(sample_dirs, desc="Merging labels"):
        sample_name = os.path.basename(sample_dir.rstrip('/')).lower()  
        print(f"Checking sample: {sample_name}")
        
        matched = False
        for organ_key, merge_group in MERGE_PAIRS.items():
            if organ_key in sample_name:
                print(f"Matched organ: {organ_key}")
                merge_labels(sample_dir, merge_group, f"{organ_key}_combined")
                matched = True
                break  
        if not matched:
            print(f"No matching organ found for {sample_name}, skipping...")

if __name__ == "__main__":
    main()