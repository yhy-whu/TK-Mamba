import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import clip 
import warnings
import json
warnings.filterwarnings("ignore")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import load_decathlon_datalist, decollate_batch, DistributedSampler
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from model.Universal_model import Universal_model
from dataset.dataloader import get_loader
from utils import loss
from utils.utils import dice_score, check_data, TEMPLATE, get_key, NUM_CLASS,ORGAN_NAME
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


torch.multiprocessing.set_sharing_strategy('file_system')


def train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE, text_features):
    model.train()
    loss_bce_ave = 0
    loss_dice_ave = 0
    loss_contrast_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y, name = batch["image"].to(args.device), batch["post_label"].float().to(args.device), batch['name']
        logit_map,projected_feature = model(x)

        term_seg_Dice = loss_seg_DICE.forward(logit_map, y, name, TEMPLATE)
        term_seg_BCE = loss_seg_CE.forward(logit_map, y, name, TEMPLATE)
        exist_labels = (y.sum(dim=[2, 3, 4]) > 0).float()  # shape: [B, NUM_CLASS]

        # contrast loss
        loss_contrast = compute_contrast_loss(projected_feature, text_features, exist_labels)
        loss = term_seg_BCE + term_seg_Dice+ loss_contrast
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f, contrast_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item(), loss_contrast.item())
        )
        loss_bce_ave += term_seg_BCE.item()
        loss_dice_ave += term_seg_Dice.item()
        loss_contrast_ave += loss_contrast.item()
        torch.cuda.empty_cache()
    print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f, ave_contrast_loss=%2.5f' % (
        args.epoch, loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator), loss_contrast_ave/len(epoch_iterator)))
    
    return loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator), loss_contrast_ave/len(epoch_iterator)

def compute_contrast_loss(projected_feature, text_features, exist_labels):
    """
    Args:
        projected_feature: [B, 512]
        text_features: [NUM_CLASS, 512]
        exist_labels: [B, NUM_CLASS]
    """

    projected_feature = F.normalize(projected_feature, dim=1)
    text_features = F.normalize(text_features, dim=1)

    logits = torch.matmul(projected_feature, text_features.t())  # [B, NUM_CLASS]
    
    loss_contrast_fn = torch.nn.BCEWithLogitsLoss()
    loss_contrast = loss_contrast_fn(logits, exist_labels)

    return loss_contrast  
    

def split_text_by_tokens(text, max_length=77):
    # split by "."
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    final_chunks = []

    for sentence in sentences:
        tokens = clip.tokenize([sentence], truncate=False).squeeze(0)
        if len(tokens) <= max_length:
            final_chunks.append(sentence)
        else:
            # split by ";"
            sub_parts = [p.strip() + ';' for p in sentence.split(';') if p.strip()]
            for part in sub_parts:
                tokens = clip.tokenize([part], truncate=False).squeeze(0)
                if len(tokens) <= max_length:
                    final_chunks.append(part)
                else:
                    # split by ","
                    sub_sub_parts = [p.strip() + ',' for p in part.split(',') if p.strip()]
                    for sub_part in sub_sub_parts:
                        tokens = clip.tokenize([sub_part], truncate=False).squeeze(0)
                        if len(tokens) <= max_length:
                            final_chunks.append(sub_part)
                        else:
                            # split by word
                            words = sub_part.split()
                            current_chunk = []
                            for word in words:
                                temp_chunk = " ".join(current_chunk + [word])
                                tokens = clip.tokenize([temp_chunk], truncate=False).squeeze(0)
                                if len(tokens) <= max_length:
                                    current_chunk.append(word)
                                else:
                                    if current_chunk:
                                        final_chunks.append(" ".join(current_chunk))
                                    current_chunk = [word]
                            if current_chunk:
                                final_chunks.append(" ".join(current_chunk))
    return final_chunks    
 
def process(args):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = int(os.environ["LOCAL_RANK"])
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)
    print(f"Rank {rank} using device {args.device}, CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")

    clip_model, _ = clip.load("ViT-B/32", device=args.device)


    with open('/root/autodl-tmp/TK_Mamba/text_description/text_descriptions.json', 'r') as f:
        text_descriptions = json.load(f)


    texts = [text_descriptions[organ] for organ in ORGAN_NAME]
    max_context_length = 77 

    split_texts = []
    for text in texts:
        split_sentences = split_text_by_tokens(text, max_context_length)
        split_texts.append(split_sentences)


    all_text_features = []
    for sentences in split_texts:
        text_tokens = clip.tokenize(sentences).to(args.device) 
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens).float() 
            all_text_features.append(text_features)


    final_text_features = [torch.mean(features, dim=0) for features in all_text_features]
    text_features = torch.stack(final_text_features)  # [NUM_CLASS, 512]
    # prepare the 3D model
    model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    backbone=args.backbone,
                    encoding=args.trans_encoding
                    )

    #Load pre-trained weights
    if args.pretrain is not None:
        model.load_params(torch.load(args.pretrain)["state_dict"])
        
    if args.trans_encoding == 'word_embedding':
        word_embedding = torch.load(args.word_embedding)
        model.organ_embedding.data = word_embedding.float()
        print('load word embedding')

    model.to(args.device)
    model.train()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[rank])
    for i in range(torch.cuda.device_count()):
        print(f"Rank {rank} GPU {i}: {torch.cuda.get_device_name(i)}, Device ID: {torch.cuda.device(i)}")
    # criterion and optimizer
    # loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_seg_DICE = loss.DiceLoss(num_classes=NUM_CLASS, ignore_index=args.ignore_index).to(args.device)
    loss_seg_CE = loss.Multi_BCELoss(num_classes=NUM_CLASS, ignore_index=args.ignore_index).to(args.device)
    if args.backbone == 'unetpp':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,
                              nesterov=False, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

    # 恢复训练
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')  
        model_dict = checkpoint['net']  
        state_dict = model.state_dict()  
    
        adjusted_dict = {}
        for key in model_dict.keys():
            if key.startswith("module.") and not list(state_dict.keys())[0].startswith("module."):
                new_key = key[7:]
            elif not key.startswith("module.") and list(state_dict.keys())[0].startswith("module."):
                new_key = "module." + key
            else:
                new_key = key

            if new_key in state_dict:
                adjusted_dict[new_key] = model_dict[key]
            else:
                print(f"Warning: Skipping key '{new_key}' not found in model.")
    
        model.load_state_dict(adjusted_dict, strict=False)  
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
    
        print('Successfully resumed from', args.resume)
    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler = get_loader(args)

    if rank == 0:
        writer = SummaryWriter(log_dir='/root/autodl-tmp/TK_Mamba/log/' + args.log_name)
        print('Writing Tensorboard logs to ', '/root/autodl-tmp/TK_Mamba/log/' + args.log_name)

    while args.epoch < args.max_epoch:
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
        scheduler.step()

        loss_dice, loss_bce,loss_contrast = train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE, text_features)
        if rank == 0:
            writer.add_scalar('train_dice_loss', loss_dice, args.epoch)
            writer.add_scalar('train_bce_loss', loss_bce, args.epoch)
            writer.add_scalar('train_contrast_loss', loss_contrast, args.epoch)
            writer.add_scalar('lr', scheduler.get_lr(), args.epoch)

        if (args.epoch % args.store_num == 0 and args.epoch != 0) and rank == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": args.epoch
            }
            if not os.path.isdir(args.model_save_path + args.log_name):
                os.mkdir(args.model_save_path + args.log_name)
            torch.save(checkpoint, args.model_save_path + args.log_name + '/epoch_' + str(args.epoch) + '.pth')
            print('save model success')

        args.epoch += 1
    if args.dist :
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank",type=int, default=0 )
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=1)
    ## logging
    parser.add_argument('--log_name', default='TK_Mamba', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--backbone', default='K_Mamba', help='backbone [K_Mamba or swinunetr or unet or dints or unetpp]')
    parser.add_argument('--ignore_index', type=int, default=11, help='Index of class to ignore in loss computation')
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default=None,  #swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
                        help='The path of pretrain model. Eg, ./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt')
    parser.add_argument('--trans_encoding', default='word_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='/root/autodl-tmp/TK_Mamba/pretrained_weights/txt_encoding.pth', 
                        help='The path of word embedding')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=3000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=50, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=50, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_10']) 
    parser.add_argument('--model_save_path', default='/root/autodl-tmp/TK_Mamba/model_result_', help='the path of saving model')
    parser.add_argument('--data_root_path', default='/root/autodl-tmp/data/', help='data root path')
    parser.add_argument('--data_txt_path', default='/root/autodl-tmp/TK_Mamba/dataset/dataset_list/', help='data txt path')
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--num_workers', default=48, type=int, help='workers numebr for DataLoader')
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
    parser.add_argument('--num_samples', default=2, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='train', help='train or validation or test')
    parser.add_argument('--uniform_sample', action="store_true", default=False, help='whether utilize uniform sample strategy')
    parser.add_argument('--datasetkey', nargs='+', default=['10_03','10_06','10_07','10_08','10_10', '10_11'],
                                            help='the content for ')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=1.0, type=float, help='The percentage of cached data in total')

    args = parser.parse_args()
    
    process(args=args)

if __name__ == "__main__":
    main()
