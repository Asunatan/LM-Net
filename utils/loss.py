import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss
class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None,n_classes=2, size_average=True,e=1e-7):
        super(BCEDiceLoss, self).__init__()
        self.e=e
        self.n_classes=n_classes

    def forward(self, pred, mask):
        batch_size, channel, _, _ = pred.size()

        # BCE_loss
        bce_loss = nn.CrossEntropyLoss()(pred, mask.long())

        # Dice_loss
        pred = torch.softmax(pred, dim=1).contiguous().view(batch_size, self.n_classes, -1)
        mask = F.one_hot(mask.long(), num_classes=self.n_classes).contiguous().permute(0, 3, 1, 2).view(batch_size,
                                                                                                        self.n_classes,
                                                                                                        -1)
        inter = torch.sum(pred * mask, 2)
        union = torch.sum(pred, 2) + torch.sum(mask, 2)
        dice = (2.0 * inter + self.e) / (union + self.e)
        dice = dice.mean()

        return 0.4*bce_loss + 0.6*(1 - dice)
def focal_loss(logits, labels, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bce_loss

    weighted_loss = alpha * loss
    loss = torch.sum(weighted_loss)
    loss /= torch.sum(labels)
    return loss
class ClassBalancedLoss(torch.nn.Module):
    def __init__(self, samples_per_class=None, num_classes=2, label_smoothing=False, beta=0.9999, gamma=2,mixup=None,
                 loss_type="focal"):
        super(ClassBalancedLoss, self).__init__()
        if loss_type not in ["focal", "sigmoid", "softmax"]:
            loss_type = "focal"
        if samples_per_class is None:
            samples_per_class = [1] * num_classes
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / (np.array(effective_num) + float("1e-8"))
        self.constant_sum = len(samples_per_class)#总类别数
        weights = (weights / np.sum(weights) * self.constant_sum).astype(np.float32)

        self.class_weights = weights
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.mixup = mixup
        if self.mixup :
            self.is_smoothing = False
        else :
            self.is_smoothing = label_smoothing

    def update(self, samples_per_class):#这里有个问题，如果全部抽到一类的怎么解决的？这个问题还大概率出现在二分类任务上
        samples_per_class=samples_per_class.cpu().numpy()
        if samples_per_class is None:
            return
        effective_num = 1.0 - np.power(self.beta, samples_per_class)
        weights = (1.0 - self.beta) / np.array(effective_num)
        self.constant_sum = len(samples_per_class)
        weights = (weights / np.sum(weights) * self.constant_sum).astype(np.float32)
        self.class_weights = weights

    def specific_smoothing(self, label, smoothing_rate=0.9):
        """
        para: label one_hot [N, num_cls]
        """
        num_class = label.shape[1]
        cls_foronebatch = label.argmax(dim=1)  # [N, 1]
        for i, cls_id in enumerate(cls_foronebatch):
            label_onehot = torch.zeros(1, num_class) * (1 - smoothing_rate) / (num_class - 1)
            label_onehot[:, cls_id] = smoothing_rate
            label[i] = label_onehot
        return label

    def forward(self, x, y):
        _, num_classes = x.shape
        if len(y.shape)>1:
            labels_one_hot=y
            weights = torch.tensor(self.class_weights, device=x.device).index_select(0, y.argmax(1))
        else:
            labels_one_hot = F.one_hot(y, num_classes).float()
            labels_one_hot = self.specific_smoothing(labels_one_hot)
            weights = torch.tensor(self.class_weights, device=x.device).index_select(0, y)
        #weights = torch.tensor(self.class_weights, device=x.device).index_select(0, y.argmax(1))

        weights = weights.unsqueeze(1)

        if self.loss_type == "focal":
            cb_loss = focal_loss(x, labels_one_hot, weights, self.gamma)
            # cb_loss = self.loss(x)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(x, labels_one_hot, weights)
        else:  # softmax
            pred = x.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(pred, labels_one_hot, weights)
        return cb_loss
class FocalLoss(nn.Module):
    def __init__(self, class_weight=None,gamma=2.0, alpha=0.5,num_classes=2):
        super(FocalLoss, self).__init__()
        self.gamma=gamma
        self.alpha=alpha
        self.num_classes=num_classes
    def forward(self, pred, target ):
        loss=0.
        for c in range(self.num_classes):
            target_c = (target == c).long().float()
            pred_c = pred[:, c, ...]
            loss += sigmoid_focal_loss(pred_c, target_c,reduction='mean')
        # n, c, h, w = pred.size()
        # criterion = nn.CrossEntropyLoss()
        # criterion = criterion.cuda()
        # logpt = -criterion(pred, target.long())
        # pt = torch.exp(logpt)
        # if self.alpha is not None:
        #     logpt *= self.alpha
        # loss = -((1 - pt) ** self.gamma) * logpt
        # if self.batch_average:
        #     loss /= n
        return loss

# class DiceLoss(nn.Module):
#     def __init__(self, weight=None,n_classes=2, size_average=True,e=1e-7):
#         super(DiceLoss, self).__init__()
#         self.e=e
#         self.n_classes=n_classes
#
#     def forward(self, pred, mask):
#         batch_size, channel, _, _ = pred.size()
#         # Dice_loss
#         pred = torch.softmax(pred, dim=1).contiguous().view(batch_size, self.n_classes, -1)
#         mask = F.one_hot(mask.long(), num_classes=self.n_classes).contiguous().permute(0, 3, 1, 2).view(batch_size,
#                                                                                                         self.n_classes,
#                                                                                                         -1)
#         inter = torch.sum(pred * mask, 2)
#         union = torch.sum(pred, 2) + torch.sum(mask, 2)
#         dice = (2.0 * inter + self.e) / (union + self.e)
#         dice = dice.mean()
#
#         return 1 - dice

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score[ignore != 1] * target[ignore != 1])
        y_sum = torch.sum(target[ignore != 1] * target[ignore != 1])
        z_sum = torch.sum(score[ignore != 1] * score[ignore != 1])
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True, ignore=None):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

####################################################################################################3

def dice_loss(pred,target,valid_mask,smooth=1,exponent=2,class_weight=None,ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                smooth=smooth,
                exponent=exponent)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss
    return total_loss / num_classes


def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwargs):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    return 1 - num / den

class offical_DiceLoss(nn.Module):

    def __init__(self,smooth=1,exponent=2,reduction='mean',class_weight=None,loss_weight=1.0,
                 ignore_index=255,loss_name='loss_dice',**kwargs):
        super(offical_DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = None
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,pred,target,avg_factor=None,reduction_override=None,**kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(torch.clamp(target.long(), 0, num_classes - 1),num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * dice_loss(pred,one_hot_target,valid_mask=valid_mask,smooth=self.smooth,exponent=self.exponent,class_weight=class_weight,ignore_index=self.ignore_index)
        return loss.sum()

































