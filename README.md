# TK_Mamba: Medical Image Segmentation with Mamba

![Method](https://github.com/user-attachments/assets/c48c4208-cb36-4892-a664-e8d249dee8f4)
This repository contains the code for the TK_Mamba project, designed for medical image segmentation tasks. The code supports training and evaluation on multiple datasets, including the Medical Segmentation Decathlon (MSD) and a modified KiTS23 dataset.

## Datasets

The project uses the following datasets:

- **Medical Segmentation Decathlon (MSD)**: Download from [Google Drive](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2). The datasets used in this project include:
  - Task03_Liver: `/root/autodl-tmp/data/10_Decathlon/Task03_Liver`
  - Task06_Lung: `/root/autodl-tmp/data/10_Decathlon/Task06_Lung`
  - Task07_Pancreas: `/root/autodl-tmp/data/10_Decathlon/Task07_Pancreas`
  - Task08_HepaticVessel: `/root/autodl-tmp/data/10_Decathlon/Task08_HepaticVessel`
  - Task10_Colon: `/root/autodl-tmp/data/10_Decathlon/Task10_Colon`
- **KiTS23 (Modified)**: Download from [KiTS23 GitHub](https://github.com/neheller/kits23). For consistency with the MSD dataset format, the KiTS23 dataset has been renamed to `Task11_kits23` and is located at `/root/autodl-tmp/data/10_Decathlon/Task11_kits23`.

### Data Format

The datasets are expected to follow the Medical Segmentation Decathlon (MSD) format, with the modified KiTS23 dataset (`Task11_kits23`) preprocessed to match this structure. Each dataset should have the following organization:

- **Folder Structure**:

  ```
  /root/autodl-tmp/data/10_Decathlon/TaskXX_<DatasetName>/
  ├── imagesTr/               # images
  │   ├── <dataset>_<id>.nii.gz  # e.g., liver_001.nii.gz, kits23_001.nii.gz
  │   └── ...
  ├── labelsTr/               # labels
  │   ├── <dataset>_<id>.nii.gz  # e.g., liver_002.nii.gz, kits23_002.nii.gz
  │   └── ...
  ├── dataset.json            # Metadata file describing the dataset
  ```

- **File Format**:

  - **Images**: 3D medical images in NIfTI format (`.nii.gz`), typically containing CT scans.
  - **Labels**: Segmentation masks in NIfTI format (`.nii.gz`), with integer values representing different classes (e.g., 0 for background, 1 for organ/tumor, and additional integers for other classes if applicable).
  - **Metadata**: A `dataset.json` file containing dataset-specific information, such as modality (e.g., CT, MRI), class labels, and data splits (training/validation). Refer to the [MSD documentation](http://medicaldecathlon.com/) for the exact schema.

- **KiTS23 Modification**:

  - The KiTS23 dataset has been renamed to `Task11_kits23` and restructured to match the MSD format (e.g., `imagesTr/kits23_1.nii.gz` for images and `labelsTr/kits23_1.nii.gz` for labels).
  - Run `label_transfer.py` to standardize label formats across datasets, ensuring compatibility with the model's expected input.

## Code Structure

The codebase is located at `/root/autodl-tmp/TK_Mamba`. Key scripts include:

- `label_transfer.py`: Converts dataset labels to a unified format compatible with the model.
- `train.py`: Trains the segmentation model. Supports distributed training with `CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore -m torch.distributed.launch --nproc_per_node=4 train.py --dist True` .
- `validation.py`: Evaluates model performance on the validation set.
- `pred_pseudo.py`: Visualizes model predictions.
- `merge_label.py`: Merges labels for the same organ across datasets (optional).

## Getting Started

To reproduce the results, follow these steps:

### Prerequisites

- Python 3.10.8
- PyTorch 2.5.1+cu124 (with distributed training support for multi-GPU setups)
- torchvision 0.20.1+cu124
- Other dependencies (list in `requirements.txt`, if applicable)

### Setup

1. **Download Datasets**:

   - Download and extract the MSD datasets from the provided [Google Drive link](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2).
   - Download the KiTS23 dataset from [GitHub](https://github.com/neheller/kits23) and place it in `/root/autodl-tmp/data/10_Decathlon/Task11_kits23` after renaming to match the MSD format.

2. **Prepare Data**:

   - Ensure the datasets follow the folder structure and file format described above.

   - Run the label conversion script to standardize the label format:

     ```bash
     python /root/autodl-tmp/TK_Mamba/label_transfer.py
     ```

3. **Train the Model**:

   - For single-GPU training:

     ```bash
     python /root/autodl-tmp/TK_Mamba/train.py
     ```

   - For distributed training (multi-GPU):

     ```bash
     CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore -m torch.distributed.launch --nproc_per_node=4 train.py --dist True
     ```

3. **Evaluate the Model**:

   - Run the validation script to test model performance:

     ```bash
     python /root/autodl-tmp/TK_Mamba/validation.py
     ```

4. **Visualize Results**:

   - Generate visualizations of model predictions:

     ```bash
     python /root/autodl-tmp/TK_Mamba/pred_pseudo.py
     ```
	- (Optional) Combine the labels of an organ and its corresponding tumor into a single label, merging their visualizations：

     ```bash
     python /root/autodl-tmp/TK_Mamba/merge_label.py
     ```

## Notes

- **Dataset Paths**: Ensure the dataset paths match the structure provided above or adjust the paths in the scripts accordingly.
- **Data Integrity**: Verify that NIfTI files are correctly formatted and accessible. Use tools like `nibabel` to inspect files if needed.
- **Distributed Training**: For distributed training, confirm that your environment supports `torchrun` and has multiple GPUs available.
- **Troubleshooting**: If you encounter issues, check the dataset integrity (e.g., file corruption, missing `dataset.json`) and ensure all dependencies are installed.

## Contact

For questions or support, please open an issue in this repository.
