import sys

#from mmcls.core import f1_score
import cv2
import numpy
import numpy as np
from skimage.metrics import hausdorff_distance

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from tqdm import tqdm
import torch
from .distributed_utils import reduce_value,get_rank

def ravd(result, reference):
    result = numpy.atleast_1d(result)
    reference = numpy.atleast_1d(reference)

    vol1 = numpy.count_nonzero(result)
    vol2 = numpy.count_nonzero(reference)

    if 0 == vol2:
        return 0
        #raise RuntimeError('The second supplied array does not contain any binary object.')

    return (vol1 - vol2) / float(vol2)

class RVDEvaluator:

    def __init__(self):
        self.pred_vol_sum = 0
        self.target_vol_sum = 0
        self.num_samples = 0

    def update(self, pred, target):
        pred_vol = torch.sum(pred)
        target_vol = torch.sum(target)
        #if target_vol != 0:
        self.pred_vol_sum += pred_vol
        self.target_vol_sum += target_vol
        self.num_samples += 1

    def get_score(self):
        if self.target_vol_sum > 0:
            rvd = (self.pred_vol_sum - self.target_vol_sum) / self.target_vol_sum
        else:
            rvd = 0
        return rvd

    def reset(self):
        self.pred_vol_sum = 0
        self.target_vol_sum = 0
        self.num_samples = 0


class Evaluator(object):
    def __init__(self, num_class=2):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
    def Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc
    def Mean_Accuracy(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum()
        Acc = np.nanmean(Acc)
        return Acc
    def Recall(self):
        return self.confusion_matrix[1][1] / (self.confusion_matrix[1][1] + self.confusion_matrix[1][0])
    def Mean_Recall(self):
        recall = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        recall = np.nanmean(recall)
        return recall
    def Precision(self):
        return self.confusion_matrix[1][1] / (self.confusion_matrix[1][1] + self.confusion_matrix[0][1])

    def Specificity(self):
        return self.confusion_matrix[0][0] / (self.confusion_matrix[0][0] + self.confusion_matrix[0][1])
        # Dice系数
    def Dice(self):
        dice_coef = (2 * self.confusion_matrix[1][1]) / (
                2 * self.confusion_matrix[1][1]+self.confusion_matrix[0][1]+self.confusion_matrix[1][0])
        dice_coef=np.nanmean(dice_coef)
        return dice_coef
    def Mean_Dice(self):
        dice_coef = (2 * np.diag(self.confusion_matrix)) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0))
        dice_coef=np.nanmean(dice_coef)
        return dice_coef


    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def train_one_epoch(model, optimizer,metric_collection=None,num_classes=2, data_loader=None,
                    device=0, criterion=None,scaler=None,criterion_dice=None):
    model.train()
    metric_collection.reset()
    # val_eval=Evaluator()
    # val_eval.reset()
    total_loss = 0.
    data_loader = tqdm(data_loader, desc=' train',file=sys.stdout)
    for images, labels in data_loader:
        images,labels=images.to(device,non_blocking=True),labels.to(device,non_blocking=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(images)#pytorch的库要加['out']
                loss = criterion(output, labels)+criterion_dice(output, labels.unsqueeze(1).float(),weight=[1.0,4.0])
                #loss = criterion(output, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(images)#['out']#pytorch的库要加['out']
            loss = criterion(output, labels)+criterion_dice(output, labels.unsqueeze(1).float(),weight=[1.0,4.0])
            #loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            data_loader.set_description('train_loss:{:.6f}'.format(loss.item()))
            total_loss+=loss.item()
            #pred = torch.softmax(output, 1)
            pred=output.argmax(1)
            #pred=pred_float.argmax(1)
            #train_pred.extend(pred_float.detach().cpu().numpy())#sklearn计算
            #train_label.extend(labels.detach().cpu().numpy())
            pred=pred.detach().cpu()
            labels=labels.detach().cpu()
            metric_collection.update(pred,labels)
    #         val_eval.add_batch(labels.numpy(), pred.numpy())
    # val_acc = val_eval.Accuracy()
    # val_precision = val_eval.Precision()
    # val_recall = val_eval.Recall()
    # val_specificity = val_eval.Specificity()
    # val_dice = val_eval.Mean_Dice()
    # val_iou = val_eval.Mean_Intersection_over_Union()
    # print('---,val_acc:{},val_prec:{},val_recall:{},val_specificity:{},val_dice:{},val_IoU:{}'.format( val_acc,val_precision, val_recall,val_specificity,
    #                                                                                           val_dice, val_iou))
    return total_loss

@torch.no_grad()
def evaluate(model, data_loader, device,metric_collection,num_classes=2,criterion=None,scaler=None):
    model.eval()
    metric_collection.reset()
    # rvd_eval=RVDEvaluator()
    # rvd_eval.reset()
    # val_pred = []
    # val_label = []
    # 用于存储预测正确的样本个数
    total_loss=0.
    hausdorff_distance_list=[]
    rvd_list=[]
    data_loader = tqdm(data_loader, desc='val', file=sys.stdout)
    for  images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        # ,non_blocking=True
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, labels)
        else:
            output = model(images)#['out']
            loss = criterion(output, labels)
        total_loss = total_loss + loss.item()
        data_loader.set_description('val_loss: {:.4f} '.format(loss.item()))
        #pred = torch.softmax(output, 1)
        pred = output.argmax(1)
        pred = pred.detach().cpu()
        labels = labels.detach().cpu()
        metric_collection.update(pred, labels)

    return total_loss,#np.mean(rvd_list)  # ,val_pred,val_label



blue   = [0,0,255] # aorta
green  = [0,255,0]    # gallbladder
red    = [255,0,0]
@torch.no_grad()
def visualization(model, data_loader, device,save_path):
    model.eval()
    data_loader = tqdm(data_loader, desc='visualization', file=sys.stdout)
    for step, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.long().to(device)

        output = model(images)
        pre=np.array(torch.softmax(output,1).argmax(1).detach().squeeze(0).cpu()).astype('uint8')
        pred=cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
        original_img= np.array(images.detach().squeeze(0).cpu())
        original_img = np.where(pred == 1, np.full_like(original_img, blue), original_img)
        original_img = np.where(pred == 2, np.full_like(original_img, green), original_img)
        original_img = np.where(pred == 3, np.full_like(original_img, red), original_img)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, original_img)



