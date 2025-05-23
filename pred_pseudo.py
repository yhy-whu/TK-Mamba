import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time

from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from model.Universal_model import Universal_model
from dataset.dataloader import get_loader_without_gt
from utils import loss
from utils.utils import dice_score, threshold_organ, visualize_label, merge_label, get_key
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS
from utils.utils import organ_post_process, threshold_organ, save_results

torch.multiprocessing.set_sharing_strategy('file_system')


def validation(model, ValLoader, val_transforms, args):
    if not os.path.exists(args.result_save_path):
        os.makedirs(args.result_save_path)
    model.eval()
    for index, batch in enumerate(tqdm(ValLoader)):
        image, name = batch["image"].cuda(), batch["name"]
        with torch.no_grad():
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.5, mode='gaussian')
            pred_sigmoid = F.sigmoid(pred)
        
        pred_hard = threshold_organ(pred_sigmoid)
        pred_hard = pred_hard.cpu()
        torch.cuda.empty_cache()

        # use organ_list to indicate the saved organ
        #organ_list = [i for i in range(0,12)]
        organ_list =[]
        if 'liver' in name[0]:
            organ_list = [1,6]
        elif 'hepaticvessel' in name[0]:
            organ_list = [3, 9]
        elif 'pancreas' in name[0]:
            organ_list = [2, 8]
        elif 'colon' in name[0]:
            organ_list = [10]
        elif 'lung' in name[0]:
            organ_list = [7]
        elif 'kits' in name[0]:
            organ_list = [4, 5, 11]

        pred_hard_post = organ_post_process(pred_hard.numpy(), organ_list, args.log_name+'/'+name[0].split('/')[0]+'/'+name[0].split('/')[-1],args)
        pred_hard_post = torch.tensor(pred_hard_post)
        batch['results'] = pred_hard_post

        save_results(batch, args.result_save_path, val_transforms, organ_list)
            
        torch.cuda.empty_cache()



def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='K_Mamba', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--data_txt_path', default='/root/autodl-tmp/TK_Mamba/dataset/dataset_list/', help='data txt path')
    parser.add_argument('--resume', default='/root/autodl-tmp/TK_Mamba/model_result_SegMamba/epoch_1200.pth', help='The path resume from checkpoint')
    parser.add_argument('--backbone', default='K_Mamba', help='backbone [K_Mamba or swinunetr or unet]')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')

    ## dataset
    parser.add_argument('--data_root_path', default='/root/autodl-tmp/data/', help='data root path')
    parser.add_argument('--result_save_path', default="/root/autodl-tmp/TK_Mamba/pre_result/", help='path for save result')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type= float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='validation', help='train or validation or test')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')

    parser.add_argument('--threshold_organ', default='Pancreas Tumor')
    parser.add_argument('--threshold', default=0.6, type=float)

    args = parser.parse_args()

    # prepare the 3D model
    model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    backbone=args.backbone,
                    encoding='word_embedding'
                    )
    
    #Load pre-trained weights
    store_dict = model.state_dict()
    checkpoint = torch.load(args.resume)
    load_dict = checkpoint['net']
    new_load_dict = {}
    for key, value in load_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]  
            new_load_dict[new_key] = value
        else:
            new_load_dict[key] = value

    num_count = 0
    for key, value in new_load_dict.items():
        if key in store_dict:
            store_dict[key] = value
            num_count += 1
        else:
            print(f"Skipping key {key} as it is not found in model state_dict")

    model.load_state_dict(store_dict, strict=True) 
    print('Use pretrained weights. Loaded', num_count, 'params into', len(store_dict.keys()), 'model parameters')

    model.cuda()

    torch.backends.cudnn.benchmark = True

    test_loader, val_transforms = get_loader_without_gt(args)

    validation(model, test_loader, val_transforms, args)

if __name__ == "__main__":
    main()
