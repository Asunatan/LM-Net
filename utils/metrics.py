import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
def get_multi_ConfusionMatrix(truth, pred, n_classes=2, dtype=np.float, **kwargs):
    batch=truth.shape[0]
    tp, tn, fp, fn=np.zeros((batch,n_classes)),np.zeros((batch,n_classes)),np.zeros((batch,n_classes)),np.zeros((batch,n_classes))
    for c in range(n_classes):
        tp[:, c], tn[:, c], fp[:, c], fn[:, c]= ConfusionMatrix(truth, pred, c=c, **kwargs)
    return tp, tn, fp, fn
def multi_eval(truth, pred, n_classes=2):

    tp, tn, fp, fn=get_multi_ConfusionMatrix(truth, pred, n_classes)
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)
    dice=2 * tp / (2 * tp + fp + fn)
    iou=tp / (tp + fp + fn)
    acc=np.nanmean(acc,1).sum()
    prec = np.nanmean(prec,1).sum()
    recall = np.nanmean(recall,1).sum()
    dice = np.nanmean(dice,1).sum()
    iou = np.nanmean(iou,1).sum()
    return acc,prec,recall,dice,iou
def ConfusionMatrix(truth, pred, c=1, dtype=np.float, **kwargs):
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    # Compute Confusion Matrix
    tp = np.logical_and(pd, gt).sum((1, 2))
    tn = np.logical_and(not_pd, not_gt).sum((1, 2))
    fp = np.logical_and(pd, not_gt).sum((1, 2))
    fn = np.logical_and(not_pd, gt).sum((1, 2))
    # Convert to desired numpy type to avoid overflow
    tp = tp.astype(dtype)
    tn = tn.astype(dtype)
    fp = fp.astype(dtype)
    fn = fn.astype(dtype)
    return tp, tn, fp, fn

def my_accuracy(tp, tn, fp, fn,e: float = 1e-7, **kwargs):
    # Obtain confusion mat
    # Calculate Accuracy
    acc = (tp + tn) / (tp + tn + fp + fn)
    # Return computed Accuracy
    return acc.sum()

def my_precision(tp, tn, fp, fn, c=1, e: float = 1e-7,**kwargs):
    # Calculate precision
    fenzi=tp
    fenmu=tp + fp
    #prec=fenzi/fenmu
    prec = np.divide(fenzi, fenmu, out=np.ones_like(fenzi), where=fenmu != 0.  )
    # Return precision
    return prec.sum()
def my_recall(tp, tn, fp, fn, c=1, e: float = 1e-7,**kwargs):
    fenzi=tp
    fenmu = tp + fn
    #sens=fenzi/fenmu
    sens = np.divide(fenzi, fenmu, out=np.ones_like(fenzi), where= fenmu != 0. )
    return sens.sum()
def my_dice(tp, tn, fp, fn, c=1, e: float = 1e-7,**kwargs):
    fenzi = 2 * tp
    fenmu = 2 * tp + fp + fn
    #dice =fenzi/fenmu
    dice = np.divide(fenzi, fenmu, out=np.ones_like(fenzi), where= fenmu != 0.)
    # Return computed Dice
    return dice.sum()

def my_iou(tp, tn, fp, fn, c=1, e: float = 1e-7,**kwargs):

    fenzi=tp
    fenmu = tp + fp + fn
    #iou = fenzi / fenmu
    iou = np.divide(fenzi, fenmu, out=np.ones_like(fenzi), where= fenmu != 0.)
    # Return computed IoU
    return iou.sum()
def binary_eval(truth, pred):
    tp, tn, fp, fn = ConfusionMatrix(truth, pred)
    acc=my_accuracy(tp, tn, fp, fn)
    prec = my_precision(tp, tn, fp, fn)
    recall = my_recall(tp, tn, fp, fn)
    dice = my_dice(tp, tn, fp, fn)
    iou = my_iou(tp, tn, fp, fn)
    return acc,prec,recall,dice,iou


# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class MetricTracker(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
def jaccard_index(input, target):

    intersection = (input * target).long().sum().data.cpu()[0]
    union = (
        input.long().sum().data.cpu()[0]
        + target.long().sum().data.cpu()[0]
        - intersection
    )

    if union == 0:
        return float("nan")
    else:
        return float(intersection) / float(max(union, 1))


# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(input, target):
    num_in_target = input.size(0)

    smooth = 1.0

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)
    intersection = (pred * truth).sum(1)

    loss = (2.0 * intersection + smooth) / (pred.sum(1) + truth.sum(1) + smooth)

    return loss.mean().item()

def iou_pytorch(predictions, labels, e: float = 1e-7):
    """Calculates Intersection over Union for a tensor of predictions"""
    predictions = torch.where(predictions > 0.5, 1, 0)
    labels = labels.byte()

    intersection = (predictions & labels).float().sum((1, 2))
    union = (predictions | labels).float().sum((1, 2))

    iou = (intersection + e) / (union + e)
    return iou


def dice_pytorch(predictions, labels, e: float = 1e-7):
    """Calculates Dice coefficient for a tensor of predictions"""
    predictions = torch.where(predictions > 0.5, 1, 0)
    labels = labels.byte()
    a=predictions & labels
    b=predictions.float().sum((1, 2))
    c= labels.float().sum((1, 2))
    intersection = (predictions & labels).float().sum((1, 2))
    return ((2 * intersection) + e) / (predictions.float().sum((1, 2)) + labels.float().sum((1, 2)) + e)


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
