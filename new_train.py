import csv
import os
import math
import tempfile
import argparse
from matplotlib import style
import matplotlib.pyplot as plt
from matplotlib import cm
from models.my_model import SwinSeg
import ignite
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torchvision.models.resnet
from torch import nn
import torch.multiprocessing as mp
import timm.data
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from dataset.data_loading import *
from seg.Deeplabv3plus import DeepLab
from seg.ESFPNet import ESFPNetStructure
from seg.FCBFormer import FCBFormer
from seg.SSformer.mit_PLD_b4 import mit_PLD_b4
from seg.TransUNet import get_r50_b16_config, VisionTransformer
from seg.TransUNet.transunet import TransUNet
from seg.att_unet import Att_unet
from seg.core.LM_Net import LM_Net
from seg.res_unet import ResUnet
from seg.res_unetplusplus import ResUnetPlusPlus
from seg.seg_former import mit_b0
from seg.uctrans_net import Config, UCTransNet
from seg.unet import Unet
from seg.unetplusplus import UNetplusplus
from swin_unet.swin_unet import SwinUnet
from utils.distributed_utils import init_distributed_mode, dist, cleanup
from utils.train_eval_utils import train_one_epoch, evaluate, visualization
from utils.loss import DiceLoss
import os
from torch.distributed import init_process_group, destroy_process_group
from utils.distributed_utils import get_rank
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC, F1Score, JaccardIndex, Dice, Specificity

#from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import hausdorff_distance
#from medpy.metric.binary import hd,hd95
from lion_pytorch import Lion
def main_single(rank, k_fold, args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    seed = args.seed + rank
    set_seed(seed)
    if args.plot:
        import matplotlib.font_manager as fm
        font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
        fp = fm.FontProperties(fname=font_path,size=48)
        # plt.rc('font', family='Times New Roman')
        fig, axs = plt.subplots(1, 3, figsize=(60, 20))
        color_list = ['r', 'g', 'b', 'y', 'm', 'c', 'k',
                      'chartreuse', 'coral', 'gold', 'lavender', 'teal']
        alpha = [1.0, 1.0, 1.0, 0.7, 1.0, 0.5,0.7,0.7,0.8,1.0,0.6,0.6]
        dataset_names=['Kvasir','Basic','BUSI']
        #unet_data = pd.read_csv(r'/home/uax/SCY/seg/unet{}_{}.csv'.format(args.dataset, k_fold))
        model_name_list=['my_unet','res50+trans_unet','unet','unet++','uctrans_net',
                                 'att_unet','res_unet', 'res_unet++', 'trans_unet', 'swin_unet'
                                 ,'deeplabv3+','FCN_ResNet50',]#'my_unet'
        for i, dataset_name in enumerate(dataset_names):
            for j,model_name in enumerate(model_name_list):
                csv_file = model_name +'{}_{}.csv'.format(dataset_name, k_fold)
                data = pd.read_csv(csv_file)
                mdice_list = data.iloc[:, -3]

                if model_name=='unet':
                    model_name='Unet'
                if model_name=='unet++':
                    model_name='Unet++'
                if model_name=='att_unet':
                    model_name='AttUnet'
                if model_name=='res_unet':
                    model_name='ResUnet'
                if model_name=='res_unet++':
                    model_name='ResUnet++'
                if model_name=='trans_unet':
                    model_name='TransUnet'
                if model_name=='res50+trans_unet':
                    model_name='R50-TransUnet'
                if model_name=='uctrans_net':
                    model_name='UCTransNet'
                if model_name=='swin_unet':
                    model_name='Swin-Unet'
                if model_name == 'deeplabv3+':
                    model_name = 'Deeplabv3+'
                if model_name == 'FCN_ResNet50':
                    model_name = 'FCN'
                if model_name=='my_unet':
                    model_name='LM-Net'

                ax = axs[i]

                ax.plot(range(0,199), mdice_list,alpha=alpha[j], label=model_name,linewidth=7)
                ax.xaxis.set_tick_params(labelsize=42)
                ax.yaxis.set_tick_params(labelsize=42)
                ax.legend(prop=fp,fontsize=40)
            if dataset_name == 'Basic':
                dataset_name = 'LGG'
            if dataset_name == 'BUSI':
                dataset_name = 'Breast Ultrasound'
            ax.set_title('Validation mDice on the {} dataset'.format(dataset_name),fontproperties=fp,fontsize=64)

        fig.supxlabel('Epochs',fontproperties=fp,fontsize=64)
        fig.supylabel('mDice',fontproperties=fp,x=-0.00001,fontsize=64)

        fig.tight_layout()
        fig.show()
        fig.savefig('Validation_mDice_curves.png')





    # 实例化训练数据集

    if args.k_fold:
        print(args.dataset)
        train_df = pd.read_csv(r'/home/uax/SCY/seg/dataset/train_{}_{}.csv'.format(args.dataset,k_fold))
        valid_df = pd.read_csv(r'/home/uax/SCY/seg/dataset/val_{}_{}.csv'.format(args.dataset,k_fold))
        test_df = pd.read_csv(r'/home/uax/SCY/seg/dataset/test_{}_0.1.csv'.format(args.dataset))
        train_dataset = MedicineDataset(train_df, 'train')
        val_dataset = MedicineDataset(valid_df, 'val')
        test_dataset = MedicineDataset(test_df, 'val')
    else:
        train_df = pd.read_csv(r'/home/uax/SCY/seg/dataset/train_{}_0.8.csv'.format(args.dataset))
        valid_df = pd.read_csv(r'/home/uax/SCY/seg/dataset/test_{}_0.2.csv'.format(args.dataset))

        train_dataset = MedicineDataset(train_df, 'train')
        val_dataset = MedicineDataset(valid_df, 'val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              drop_last=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    test_loader=DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    # 实例化模型
    if args.model=='LM_Net':
        print('LM_Net')
        model = LM_Net(3, args.num_classes, deep_supervision=args.deep_supervision)
        print(args.model)
    if args.visualization is True:
        save_path = '/home/uax/SCY/seg/visualization'
        model.to(rank)
        model.load_state_dict(torch.load(args.model + 'best_segmentation.pth')['model_state_dict'])
        visualization(model,test_loader,rank,save_path=save_path)
        print('exit')
        exit()
    if args.apm:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    if args.mixup is True:
        pass
    else:
        mixup_fn = None

    model.to(rank)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1,4.0]).float().to(rank),label_smoothing=args.smoothing)
    criterion_dice = DiceLoss(args.num_classes).cuda(rank)
    # Scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,eta_min=1e-6)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    #scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    best_iou = 0.
    metric_collection = MetricCollection([
        Accuracy(task=args.categories, num_classes=args.num_classes, average='macro'),
        Precision(task=args.categories, num_classes=args.num_classes, average='macro'),
        Recall(task=args.categories, num_classes=args.num_classes, average='macro'),
        Specificity(task=args.categories, num_classes=args.num_classes, average='macro'),
        Dice(num_classes=args.num_classes, average='macro'),
        JaccardIndex(task=args.categories, num_classes=args.num_classes, average='macro'),
        JaccardIndex(task='multiclass', num_classes=args.num_classes, average='macro'),
    ])
    metric_collection.to('cpu')
    start_epoch=0
    if args.resume:
        checkpoint = torch.load('{}{}_{}_checkpoint.pth'.format(args.model,args.dataset,k_fold))
        start_epoch=checkpoint['epoch']+1
        scheduler.load_state_dict(checkpoint['lr_schedule'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
    if args.test:
        state_dict=torch.load('/home/uax/SCY/seg/unetKvasir_0_checkpoint.pth')
        model.load_state_dict(state_dict['state_dict'])
        test_loss= evaluate(model=model, data_loader=test_loader, criterion=criterion,
                            metric_collection=metric_collection,
                            device=rank, scaler=scaler)
        test_accuracy, test_precision, test_recall, test_specificity, test_dice, test_iou, test_mean_iou = metric_collection.compute().values()
        print(" test_loss:{:.4f} test_accuracy:{:.4f} test_precision:{:.4f} test_recall:{:.4f} test_specificity:{:.4f} test_dice:{:.4f} test_iou:{:.4f} test_mean_iou:{:.4f}" \
                .format(test_loss, test_accuracy, test_precision, test_recall,test_specificity, test_dice, test_iou,test_mean_iou))
        metric_list=[
                test_loss, test_accuracy, test_precision, test_recall, test_specificity, test_dice, test_iou,test_mean_iou
                ]
        with open(args.model +args.dataset+ 'test_rvd_class.csv'.format(k_fold), 'a', encoding='utf-8', newline='') as fw:
            writer = csv.writer(fw)
            writer.writerow("{:.4f}".format(element) for element in metric_list)
        exit(0)
    for epoch in range(start_epoch,args.epochs):

        print('epoch:{}   lr0:{}'.format(epoch + 1, optimizer.state_dict()['param_groups'][0]['lr'],
                                                  ))
        train_loss = train_one_epoch(model=model, optimizer=optimizer,metric_collection=metric_collection, num_classes=args.num_classes,
                                     data_loader=train_loader, device=rank,criterion=criterion,criterion_dice=criterion_dice,
                                    scaler=scaler)
        train_accuracy, train_precision, train_recall, train_specificity, train_dice, train_iou, train_mean_iou = metric_collection.compute().values()
        val_loss = evaluate(model=model, data_loader=val_loader, criterion=criterion,
                            metric_collection=metric_collection,
                            device=rank, scaler=scaler)

        val_accuracy, val_precision, val_recall, val_specificity,val_dice, val_iou,val_mean_iou = metric_collection.compute().values()
        scheduler.step()
        print(
            " train_loss:{:.4f} train_accuracy:{:.4f} train_precision:{:.4f} train_recall:{:.4f} train_specificity:{:.4f} train_dice:{:.4f} train_iou:{:.4f}  train_mean_iou:{:.4f}"\
                .format(train_loss, train_accuracy, train_precision, train_recall,train_specificity, train_dice, train_iou,train_mean_iou))
        print(
            " val_loss:{:.4f} val_accuracy:{:.4f} val_precision:{:.4f} val_recall:{:.4f} val_specificity:{:.4f} val_dice:{:.4f} val_iou:{:.4f} val_mean_iou:{:.4f} " \
                .format(val_loss, val_accuracy, val_precision, val_recall,val_specificity, val_dice, val_iou,val_mean_iou))
        metric_list=[train_loss, train_accuracy, train_precision, train_recall,train_specificity, train_dice, train_iou,train_mean_iou,
                val_loss,val_accuracy, val_precision, val_recall,val_specificity, val_dice,val_iou,val_mean_iou
                ]

        with open(args.model +args.dataset+ '_{}.csv'.format(k_fold), 'a', encoding='utf-8', newline='') as fw:
            writer = csv.writer(fw)
            writer.writerow("{:.4f}".format(element) for element in metric_list)
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_schedule': scheduler.state_dict()}
        torch.save(checkpoint, args.model+args.dataset + '_{}_checkpoint.pth'.format(k_fold))
        if val_iou > best_iou:
            best_iou = val_iou
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict()}
            torch.save(checkpoint, args.model +args.dataset+ 'best_{}.pth'.format(k_fold))
            with open(args.model+args.dataset + 'bestresult_{}.csv'.format(k_fold), 'w', encoding='utf-8', newline='') as fw:
                writer = csv.writer(fw)
                writer.writerow("{:.4f}".format(element) for element in metric_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=False)
    parser.add_argument('--smoothing', type=float, default=0.001,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--num_workers', type=float, default=2)
    parser.add_argument('--dataset', type=str, default='Kvasir',
                        choices=['Basic','Kvasir','BUSI','CVCDataset','VOC2012'],help='choose dataset')
    parser.add_argument('--model', type=str, default='LM_Net',
                        choices=['res50+trans_unet','LM_Net','unet','unet++','uctrans_net','pre_swin_unet',
                                 'att_unet','res_unet', 'res_unet++', 'trans_unet', 'swin_unet',
                                 'seg_former','fcb_former','deeplabv3+','ssformer','FCN_ResNet50','ESFPNet'],
                        help='choose model')
    parser.add_argument('--categories', type=str, default='binary', choices=['binary', 'multiclass', 'multilabel'])
    parser.add_argument('--visualization', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--resume', type=bool, default=False, help='put the path to resuming file if needed')
    # 不要改该参数，系统会自动分配
    parser.add_argument('--rank', default=0, help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--local_rank', default=0, help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--seed', default=42, help='set seed')
    parser.add_argument('--deep_supervision', type=bool, default=False, choices=[True, False],
                        help='choose deep_supervision')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--mixup', default=False, help='mixup')
    parser.add_argument('--apm', default=False, help='Automatic Mixed Precision')
    parser.add_argument('--plot', default=False, help='distributed')
    parser.add_argument('--distributed', default=False, help='distributed')
    parser.add_argument('--k_fold', type=bool, default=True)
    arg = parser.parse_args()

    # print('DDP is available -> {}'.format(torch.distributed.is_available()))
    world_size = torch.cuda.device_count()
    if arg.k_fold:
        k_fold=5
        for i in range(0,k_fold):
            print('第{}折'.format(i))
            main_single(0, i, arg)
    else:
        main_single(0, 0, arg)
    exit()
