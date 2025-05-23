
# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sklearn.model_selection import KFold  ## K折交叉验证
import pickle
import os
import json
import math
import numpy as np
import torch
from monai import transforms
import SimpleITK as sitk
from tqdm import tqdm 
from torch.utils.data import Dataset 
import glob 
from light_training.dataloading.utils import unpack_dataset
import random 
class MedicalDataset(Dataset):
    def __init__(self, datalist, test=False) -> None:
        super().__init__()
        from train_3 import test_label_dir  # 在函数内导入，避免循环导入
        self.datalist = datalist
        self.test = test 
        self.test_label_dir = test_label_dir  # 测试集标签路径
        self.data_cached = []
        for p in tqdm(self.datalist, total=len(self.datalist)):
            info = self.load_pkl(p)
            self.data_cached.append(info)
            
        ## unpacking
        print(f"unpacking data ....")
        # for 
        folder = []
        for p in self.datalist:
            f = os.path.dirname(p)
            if f not in folder:
                folder.append(f)
        for f in folder:
            unpack_dataset(f, 
                        unpack_segmentation=True,
                        overwrite_existing=False,
                        num_processes=8)


        print(f"data length is {len(self.datalist)}")
    def load_pkl(self, data_path):
        
        pass 
        properties_path = f"{data_path[:-4]}.pkl"
        df = open(properties_path, "rb")
        info = pickle.load(df)


        return info 

    def post(self, batch_data):
        return batch_data
    
    def read_data(self, data_path):
        
        image_path = data_path.replace(".npz", ".npy")
        image_data = np.load(image_path, "r+")
        
        
        seg_data = None 
        if not self.test:
            seg_path = data_path.replace(".npz", "_seg.npy")
            seg_data = np.load(seg_path, "r+")
        else:  # 测试集从 .nii.gz 文件加载标签
            label_name = os.path.basename(data_path).replace(".npz", ".nii.gz")
            label_path = os.path.join(self.test_label_dir, label_name)
            print(f"Loading label from: {label_path}")
            seg_data = self.load_sitk_label(label_path)
        return image_data, seg_data


    def load_sitk_label(self, label_path):
        """使用 SimpleITK 加载 .nii.gz 文件并转换为 NumPy 格式。"""
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        # 使用 SimpleITK 加载 NIfTI 标签文件
        sitk_img = sitk.ReadImage(label_path)
        label_data = sitk.GetArrayFromImage(sitk_img).astype(np.int8)  # 转换为 NumPy 格式

        return label_data


    def __getitem__(self, i):
        
        image, seg = self.read_data(self.datalist[i])

        properties = self.data_cached[i]
        if seg is None:
            return {
                "data": image,
                "properties": properties
            }
        else :
            return {
                "data": image,
                "seg": seg,
                "properties": properties
            }

    def __len__(self):
        return len(self.datalist)
    
def get_train_test_loader_from_test_list(data_dir, test_list):
    all_paths = glob.glob(f"{data_dir}/*.npz")

    test_datalist = []
    train_datalist = []

    test_list_1 = []
    for t in test_list:
        test_list_1.append(t.replace(".nii.gz", ""))

    test_list = test_list_1
    for p in all_paths:
        p2 = p.split("/")[-1].split(".")[0]
        if p2 in test_list:
            test_datalist.append(p)
        else :
            train_datalist.append(p)

    print(f"training data is {len(train_datalist)}")
    print(f"test data is {len(test_datalist)}", test_datalist)

    train_ds = MedicalDataset(train_datalist)
    test_ds = MedicalDataset(test_datalist)

    loader = [train_ds, test_ds]

    return loader

def get_kfold_data(data_paths, n_splits, shuffle=False):
    X = np.arange(len(data_paths))
    kfold = KFold(n_splits=n_splits, shuffle=shuffle)  ## kfold为KFolf类的一个对象
    return_res = []
    for a, b in kfold.split(X):
        fold_train = []
        fold_val = []
        for i in a:
            fold_train.append(data_paths[i])
        for j in b:
            fold_val.append(data_paths[j])
        return_res.append({"train_data": fold_train, "val_data": fold_val})

    return return_res

def get_kfold_loader(data_dir, fold=0, test_dir=None):

    all_paths = glob.glob(f"{data_dir}/*.npz")
    fold_data = get_kfold_data(all_paths, 5)[fold]

    train_datalist = fold_data["train_data"]
    val_datalist = fold_data["val_data"]  

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    train_ds = MedicalDataset(train_datalist)
    
    val_ds = MedicalDataset(val_datalist)

    if test_dir is not None:
        test_paths = glob.glob(f"{test_dir}/*.npz")
        test_ds = MedicalDataset(test_paths, test=True)
    else:
        test_ds = None 

    loader = [train_ds, val_ds, test_ds]

    return loader

def get_all_training_loader(data_dir, fold=0, test_dir=None):
    ## train all labeled data 
    ## fold denote the validation data in training data
    all_paths = glob.glob(f"{data_dir}/*.npz")
    fold_data = get_kfold_data(all_paths, 5)[fold]

    train_datalist = all_paths
    val_datalist = fold_data["val_data"]  

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    train_ds = MedicalDataset(train_datalist)
    
    val_ds = MedicalDataset(val_datalist)

    if test_dir is not None:
        test_paths = glob.glob(f"{test_dir}/*.npz")
        test_ds = MedicalDataset(test_paths, test=True)
    else:
        test_ds = None 

    loader = [train_ds, val_ds, test_ds]

    return loader

def get_train_val_test_loader_seperate(train_dir, val_dir, test_dir=None):
    train_datalist = glob.glob(f"{train_dir}/*.npz")
    val_datalist = glob.glob(f"{val_dir}/*.npz")

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")

    if test_dir is not None:
        test_datalist = glob.glob(f"{test_dir}/*.npz")
        print(f"test data is {len(test_datalist)}")
        test_ds = MedicalDataset(test_datalist, test=True)
    else :
        test_ds = None

    train_ds = MedicalDataset(train_datalist)
    val_ds = MedicalDataset(val_datalist)
    
    loader = [train_ds, val_ds, test_ds]

    return loader

def get_train_val_test_loader_from_split_json(data_dir, split_json_file):
    import json 

    with open(split_json_file, "r") as f:

        datalist = json.loads(f.read())

    train_datalist = datalist["training"]
    val_datalist = datalist["validation"]
    test_datalist = datalist["test"]

    def add_pre(datalist):
        for i in range(len(datalist)):
            datalist[i]["image"] = os.path.join(data_dir, datalist[i]["image"].replace("imagesTr/", ""))
            datalist[i]["label"] = os.path.join(data_dir, datalist[i]["label"].replace("labelsTr/", ""))
    
    add_pre(train_datalist)
    add_pre(val_datalist)
    add_pre(test_datalist)
    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    # 根据 "image" 键进行排序，避免字典之间比较出错
    print(f"Test data: {len(test_datalist)} files", 
          sorted(test_datalist, key=lambda x: x["image"]))
    
    train_ds = MedicalDataset(train_datalist)
    val_ds = MedicalDataset(val_datalist)
    test_ds = MedicalDataset(test_datalist)

    loader = [train_ds, val_ds, test_ds]

    return loader


def get_train_val_test_loader_from_train(data_dir, train_rate=0.7, val_rate=0.1, test_rate=0.2, seed=42):
    ## train all labeled data 
    ## fold denote the validation data in training data
    all_paths = glob.glob(f"{data_dir}/*.npz")
    # fold_data = get_kfold_data(all_paths, 5)[fold]

    train_number = int(len(all_paths) * train_rate)
    val_number = int(len(all_paths) * val_rate)
    test_number = int(len(all_paths) * test_rate)
    random.seed(seed)
    # random_state = random.random
    random.shuffle(all_paths)

    train_datalist = all_paths[:train_number]
    val_datalist = all_paths[train_number: train_number + val_number]
    test_datalist = all_paths[-test_number:] 

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    print(f"test data is {len(test_datalist)}", sorted(test_datalist))

    train_ds = MedicalDataset(train_datalist)
    val_ds = MedicalDataset(val_datalist)
    test_ds = MedicalDataset(test_datalist)

    loader = [train_ds, val_ds, test_ds]

    return loader

def get_train_loader_from_train(data_dir):
    ## train all labeled data 
    ## fold denote the validation data in training data
    all_paths = glob.glob(f"{data_dir}/*.npz")
    # fold_data = get_kfold_data(all_paths, 5)[fold]

    train_ds = MedicalDataset(all_paths)
  
    return train_ds

def get_test_loader_from_test(data_dir):
    all_paths = glob.glob(f"{data_dir}/*.npz")

    test_ds = MedicalDataset(all_paths)
  
    return test_ds

def get_multi_dir_training_loader(data_dir, fold=0, test_dir=None):
    ## train all labeled data 
    ## fold denote the validation data in training data
    all_paths = []
    for p in data_dir:
        paths = glob.glob(f"{p}/*.npz")
        for pp in paths:
            all_paths.append(pp)

    # print(all_paths)
    fold_data = get_kfold_data(all_paths, 5)[fold]

    train_datalist = all_paths
    val_datalist = fold_data["val_data"]  

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    train_ds = MedicalDataset(train_datalist)
    
    val_ds = MedicalDataset(val_datalist)

    if test_dir is not None:
        test_paths = glob.glob(f"{test_dir}/*.npz")
        test_ds = MedicalDataset(test_paths, test=True)
    else:
        test_ds = None 

    loader = [train_ds, val_ds, test_ds]

    return loader

def get_loader_from_json(data_dir, json_file):
    """
    从 JSON 文件中读取数据集名称，并匹配指定路径中的文件，生成训练、验证和测试集加载器。
    """
    # 读取 JSON 文件
    with open(json_file, "r") as f:
        datalist = json.load(f)

    # 从 JSON 中提取文件名（去掉路径和扩展名）
    def extract_filename(item):
        return os.path.basename(item["image"]).split(".")[0]

    train_names = [extract_filename(item) for item in datalist["training"]]
    val_names = [extract_filename(item) for item in datalist["validation"]]
    test_names = [extract_filename(item) for item in datalist.get("test", [])]

    # 获取数据目录下所有的 npz 文件路径
    all_paths = glob.glob(f"{data_dir}/*.npz")

    # 根据文件名匹配 npz 文件路径
    def match_paths(names):
        return [p for p in all_paths if os.path.basename(p).split(".")[0] in names]

    train_datalist = match_paths(train_names)
    val_datalist = match_paths(val_names)
    test_datalist = match_paths(test_names)

    # 打印结果检查
    print(f"训练集大小: {len(train_datalist)}")
    print(f"验证集大小: {len(val_datalist)}")
    print(f"测试集大小: {len(test_datalist)}")
    

    # 创建数据集
    train_ds = MedicalDataset(train_datalist)
    val_ds = MedicalDataset(val_datalist)
    test_ds = MedicalDataset(test_datalist,test=True)

    loader = [train_ds, val_ds, test_ds]

    return loader
