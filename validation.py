import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import glob

from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from model.Universal_model import Universal_model
from dataset.dataloader import get_loader
from utils import loss
from utils.utils import dice_score, TEMPLATE, ORGAN_NAME, merge_label, visualize_label, get_key, NUM_CLASS
from utils.utils import extract_topk_largest_candidates, organ_post_process, threshold_organ

from medpy.metric.binary import __surface_distances

torch.multiprocessing.set_sharing_strategy('file_system')

def normalized_surface_dice(a: np.ndarray, b: np.ndarray, threshold: float, spacing: tuple = None, connectivity=1):
    assert all([i == j for i, j in zip(a.shape, b.shape)]), "a and b must have the same shape. a.shape= %s, " \
                                                            "b.shape= %s" % (str(a.shape), str(b.shape))
    if spacing is None:
        spacing = tuple([1 for _ in range(len(a.shape))])
    
    if a.sum() == 0 or b.sum() == 0:
        return 0.0  
    
    a_to_b = __surface_distances(a, b, spacing, connectivity)
    b_to_a = __surface_distances(b, a, spacing, connectivity)
    
    numel_a = len(a_to_b)
    numel_b = len(b_to_a)
    
    tp_a = np.sum(a_to_b <= threshold) / numel_a if numel_a > 0 else 0
    tp_b = np.sum(b_to_a <= threshold) / numel_b if numel_b > 0 else 0
    fp = np.sum(a_to_b > threshold) / numel_a if numel_a > 0 else 0
    fn = np.sum(b_to_a > threshold) / numel_b if numel_b > 0 else 0
    
    dc = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + 1e-8)
    return dc

def validation(model, ValLoader, args, i):
    model.eval()
    dice_list = {}
    nsd_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS))  # 1st row for dice, 2nd row for count
        nsd_list[key] = np.zeros((2, NUM_CLASS))   # 1st row for nsd, 2nd row for count
    
    for index, batch in enumerate(tqdm(ValLoader)):
        image, label, name = batch["image"].cuda(), batch["post_label"], batch["name"]
        try:
            spacing = batch["spacing"][0]  
        except KeyError:
            import SimpleITK as sitk
            img = sitk.ReadImage(batch["image_meta_dict"]["filename_or_obj"][0])
            spacing = img.GetSpacing()  # (x, y, z)

        with torch.no_grad():
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=args.overlap, mode='gaussian')
            pred_sigmoid = F.sigmoid(pred)
        pred_hard = threshold_organ(pred_sigmoid)
        pred_hard = pred_hard.cpu()
        torch.cuda.empty_cache()
        
        B = pred_sigmoid.shape[0]
        for b in range(B):
            content = 'case%s| '%(name[b])
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            pred_hard_post = organ_post_process(pred_hard.numpy(), organ_list, 
                                                args.log_name+'/'+name[0].split('/')[0]+'/'+name[0].split('/')[-1], args)
            pred_hard_post = torch.tensor(pred_hard_post)
            
            for organ in organ_list:
                if torch.sum(label[b, organ-1, :, :, :]) != 0:  
                    dice_organ, recall, precision = dice_score(pred_hard_post[b, organ-1, :, :, :].cuda(), 
                                                              label[b, organ-1, :, :, :].cuda())
                    dice_list[template_key][0][organ-1] += dice_organ.item()
                    dice_list[template_key][1][organ-1] += 1

                    pred_np = pred_hard_post[b, organ-1, :, :, :].numpy().astype(np.uint8)
                    label_np = label[b, organ-1, :, :, :].numpy().astype(np.uint8)
                    nsd_organ = normalized_surface_dice(pred_np, label_np, threshold=2.0, spacing=spacing)
                    nsd_list[template_key][0][organ-1] += nsd_organ
                    nsd_list[template_key][1][organ-1] += 1

                    content += '%s: Dice=%.4f, NSD=%.4f, '%(ORGAN_NAME[organ-1], dice_organ.item(), nsd_organ)
                    print('%s: Dice=%.4f, Recall=%.4f, Precision=%.4f, NSD=%.4f'%(ORGAN_NAME[organ-1], 
                                                                                 dice_organ.item(), recall.item(), 
                                                                                 precision.item(), nsd_organ))
            print(content)

        torch.cuda.empty_cache()

    ave_organ_dice = np.zeros((2, NUM_CLASS))
    ave_organ_nsd = np.zeros((2, NUM_CLASS))
    
    output_dir = f'/root/autodl-tmp/TK_Mamba/out/{args.log_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_dir+f'/b_val_{i-1}.txt', 'w') as f:
        for key in TEMPLATE.keys():
            organ_list = TEMPLATE[key]
            content_dice = 'Task%s (Dice)| '%(key)
            content_nsd = 'Task%s (NSD)| '%(key)
            for organ in organ_list:
                dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1] if dice_list[key][1][organ-1] > 0 else 0
                nsd = nsd_list[key][0][organ-1] / nsd_list[key][1][organ-1] if nsd_list[key][1][organ-1] > 0 else 0
                
                content_dice += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice)
                content_nsd += '%s: %.4f, '%(ORGAN_NAME[organ-1], nsd)
                
                ave_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
                ave_organ_dice[1][organ-1] += dice_list[key][1][organ-1]
                ave_organ_nsd[0][organ-1] += nsd_list[key][0][organ-1]
                ave_organ_nsd[1][organ-1] += nsd_list[key][1][organ-1]

            print(content_dice)
            print(content_nsd)
            f.write(content_dice + '\n')
            f.write(content_nsd + '\n')
        
        content_dice = 'Average (Dice)| '
        content_nsd = 'Average (NSD)| '
        for i in range(NUM_CLASS):
            dice_avg = ave_organ_dice[0][i] / ave_organ_dice[1][i] if ave_organ_dice[1][i] > 0 else 0
            nsd_avg = ave_organ_nsd[0][i] / ave_organ_nsd[1][i] if ave_organ_nsd[1][i] > 0 else 0
            content_dice += '%s: %.4f, '%(ORGAN_NAME[i], dice_avg)
            content_nsd += '%s: %.4f, '%(ORGAN_NAME[i], nsd_avg)
        
        print(content_dice)
        print(content_nsd)
        f.write(content_dice + '\n')
        f.write(content_nsd + '\n')

        overall_dice = np.mean([ave_organ_dice[0][i] / ave_organ_dice[1][i] for i in range(NUM_CLASS) if ave_organ_dice[1][i] > 0])
        overall_nsd = np.mean([ave_organ_nsd[0][i] / ave_organ_nsd[1][i] for i in range(NUM_CLASS) if ave_organ_nsd[1][i] > 0])
        print(f"Overall Average Dice: {overall_dice:.4f}")
        print(f"Overall Average NSD: {overall_nsd:.4f}")
        f.write(f"Overall Average Dice: {overall_dice:.4f}\n")
        f.write(f"Overall Average NSD: {overall_nsd:.4f}\n")

def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=2)
    ## logging
    parser.add_argument('--log_name', default='TK_Mamba', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--start_epoch', default=490, type=int, help='Number of start epoches')
    parser.add_argument('--end_epoch', default=490, type=int, help='Number of end epoches')
    parser.add_argument('--epoch_interval', default=100, type=int, help='Number of start epoches')
    parser.add_argument('--backbone', default='K_Mamba', help='backbone [K_Mamba or swinunetr or unet]')

    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_10']) 
    parser.add_argument('--model_save_path', default='/root/autodl-tmp/TK_Mamba/model_result_', help='the path of saving model')    
    parser.add_argument('--data_root_path', default='/root/autodl-tmp/data/', help='data root path')
    parser.add_argument('--data_txt_path', default='/root/autodl-tmp/TK_Mamba/dataset/dataset_list/', help='data txt path')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='validation', help='train or validation or test')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.05, type=float, help='The percentage of cached data in total')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--overlap', default=0.5, type=float, help='overlap for sliding_window_inference')

    args = parser.parse_args()

    # prepare the 3D model
    model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),
                            in_channels=1,
                            out_channels=NUM_CLASS,
                            backbone=args.backbone,
                            encoding='word_embedding')

    # Load pre-trained weights
    store_path_root = args.model_save_path + args.log_name +'/epoch_***.pth'
    for store_path in glob.glob(store_path_root):
        store_dict = model.state_dict()
        load_dict = torch.load(store_path)['net']

        for key, value in load_dict.items():
            if 'swinViT' in key or 'encoder' in key or 'decoder' in key:
                name = '.'.join(key.split('.')[1:])
            else:
                name = '.'.join(key.split('.')[1:])
            store_dict[name] = value

        model.load_state_dict(store_dict)
        print(f'Load {store_path} weights')

        model.cuda()
        torch.backends.cudnn.benchmark = True

        validation_loader, val_transforms = get_loader(args)
        i = int(store_path.split('_')[-1].split('.')[0]) + 1
        validation(model, validation_loader, args, i)

if __name__ == "__main__":
    main()