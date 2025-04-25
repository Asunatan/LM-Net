import logging
import os
import random
from os import listdir
from os.path import splitext
from pathlib import Path
import cv2
import pandas as pd
import torchvision
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import VOCSegmentation
from tqdm import tqdm


#data_img = r"F:\Pycharm\workshop\Segmentation\kaggle_3m//"
#data_img = r"../kaggle_3m/"
# dir_mask = Path('./data/masks/')
# dir_checkpoint = Path('./checkpoints/')
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
def create_BasicDataset_df():
    data_dir='/home/uax/SCY/seg/dataset/kaggle_3m'
    images, masks ,diagnosis=  [], [],[]
    for root, folders, files in os.walk(data_dir):  # 在目录树中游走,会打开子文件夹
        for file in files:
            if 'mask' in file:
                mask_abpath=os.path.join(root, file)
                value = np.max(cv2.imread(mask_abpath))
                if value > 0:
                    diagnosis.append(1)
                else :
                    diagnosis.append(0)
                masks.append(mask_abpath)
                images.append(mask_abpath.replace('_mask', ''))
    PathDF = pd.DataFrame({'images': images, 'masks': masks,'diagnosis':diagnosis})
    # train_df_8, test_df_2 = train_test_split(PathDF, random_state=10086, test_size=0.2,shuffle=True,stratify=PathDF['diagnosis'])
    # # train_df_8, test_df_2 = train_test_split(PathDF, random_state=10086, test_size=0.2,shuffle=True)
    # test_df_2.to_csv("test_Basic_0.2.csv", index=False)
    # train_df_8.to_csv("train_Basic_0.8.csv", index=False)
    # 5折
    train_val_df, test_df = train_test_split(PathDF, random_state=10086, test_size=0.1, shuffle=True,stratify=PathDF['diagnosis'])
    test_df.to_csv("test_Basic_0.1.csv", index=False)
    train_df, val_df = train_test_split(train_val_df, random_state=10086, test_size=0.1, shuffle=True)
    val_df.to_csv("val_Basic_0.1.csv", index=False)
    train_df.to_csv("train_Basic_0.8.csv", index=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for Fold_i, (train_index, val_index) in enumerate(skf.split(train_val_df, train_val_df['diagnosis'])):
        X_train, X_test = train_val_df.iloc[train_index], train_val_df.iloc[val_index]
        X_train.to_csv("train_Basic_{}.csv".format(Fold_i), index=False)
        X_test.to_csv("val_Basic_{}.csv".format(Fold_i), index=False)

    #valid_df, test_df = train_test_split(rest, random_state=42, test_size=0.1,shuffle=True)
    #return train_df, valid_df
def create_KvasirDataset_df():
    data_dir='/data/scy/SCY/LM-Net-main/dataset/Kvasir-SEG'
    images, masks = [], []
    for root, folders, files in os.walk(data_dir):  # 在目录树中游走,会打开子文件夹
        for file in files:
            images_abpath = os.path.join(root,file)
            images.append(images_abpath)
            masks_path=images_abpath.replace('images','masks')
            masks.append(masks_path)

    PathDF = pd.DataFrame({'images': images, 'masks': masks})
    # train_df_8, test_df_2 = train_test_split(PathDF, random_state=10086, test_size=0.2,shuffle=True)
    # test_df_2.to_csv("test_Kvasir_0.2.csv", index=False)
    # train_df_8.to_csv("train_Kvasir_0.8.csv", index=False)
    #5折
    train_val_df, test_df = train_test_split(PathDF, random_state=10086, test_size=0.1, shuffle=True)
    test_df.to_csv("test_Kvasir_0.1.csv", index=False)
    train_df, val_df = train_test_split(train_val_df, random_state=10086, test_size=0.1, shuffle=True)
    val_df.to_csv("val_Kvasir_0.1.csv", index=False)
    train_df.to_csv("train_Kvasir_0.8.csv", index=False)
    skf = KFold(n_splits=5,shuffle=True)
    for Fold_i,(train_index, val_index) in enumerate(skf.split(train_val_df)):
        X_train, X_test = train_val_df.iloc[train_index], train_val_df.iloc[val_index]
        X_train.to_csv("train_Kvasir_{}.csv".format(Fold_i), index=False)
        X_test.to_csv("val_Kvasir_{}.csv".format(Fold_i), index=False)


    #valid_df, test_df = train_test_split(valid_df, random_state=42, test_size=0.2,shuffle=True)
    #return train_df, valid_df#,test_df

def create_BUSI_Dataset_df():
    data_dir='/home/uax/SCY/seg/dataset/Breast_Ultrasound'
    images, masks ,diagnosis=  [], [],[]
    for root, folders, files in os.walk(data_dir):  # 在目录树中游走,会打开子文件夹
        for file in files:
            if '_mask' in file:
                mask_abpath=os.path.join(root,file)
                masks.append(mask_abpath)
                images.append(mask_abpath.replace('_mask', ''))
                if 'benign' in file:
                    diagnosis.append(1)
                if 'malignant' in file:
                    diagnosis.append(2)
                if 'normal' in file:
                    diagnosis.append(0)
    PathDF = pd.DataFrame({'images': images, 'masks': masks,'diagnosis':diagnosis})
    train_df_8, test_df_2 = train_test_split(PathDF, random_state=10086, test_size=0.2, shuffle=True,
                                             stratify=PathDF['diagnosis'])
    # train_df_8, test_df_2 = train_test_split(PathDF, random_state=10086, test_size=0.2,shuffle=True)
    # test_df_2.to_csv("test_BUSI_0.2.csv", index=False)
    # train_df_8.to_csv("train_BUSI_0.8.csv", index=False)
    # 5折
    train_val_df, test_df = train_test_split(PathDF, random_state=10086, test_size=0.1, shuffle=True,
                                         stratify=PathDF['diagnosis'])
    train_df, val_df = train_test_split(train_val_df, random_state=10086, test_size=0.1, shuffle=True)
    val_df.to_csv("val_BUSI_0.1.csv", index=False)
    test_df.to_csv("test_BUSI_0.1.csv", index=False)
    train_df.to_csv("train_BUSI_0.8.csv", index=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for Fold_i, (train_index, val_index) in enumerate(skf.split(train_val_df, train_val_df['diagnosis'])):
        X_train, X_test = train_val_df.iloc[train_index], train_val_df.iloc[val_index]
        X_train.to_csv("train_BUSI_{}.csv".format(Fold_i), index=False)
        X_test.to_csv("val_BUSI_{}.csv".format(Fold_i), index=False)


def create_CVC_Dataset_df(data_dir):
    dirs, images, masks ,diagnosis= [], [], [],[]
    for root, folders, files in os.walk(data_dir):  # 在目录树中游走,会打开子文件夹
        for file in files:
            dirs.append(root.replace('Original', ''))
            masks.append(os.path.join('Ground Truth',file))
            images.append(os.path.join('Original',file))
    PathDF = pd.DataFrame({'directory': dirs, 'images': images, 'masks': masks})
    train_df, valid_df = train_test_split(PathDF, random_state=42, test_size=0.2,shuffle=True)
    #train_df, valid_df = train_test_split(PathDF, random_state=42, test_size=0.2,shuffle=True)
    #valid_df, test_df = train_test_split(valid_df, random_state=42, test_size=0.2,shuffle=True)
    return train_df, valid_df#,test_df

def create_Thyroid_Dataset_df(data_dir):
    images, masks = [], []
    for file in os.listdir(data_dir):  # 在目录树中游走,会打开子文件夹
        masks.append(os.path.join(data_dir.replace('image','mask'),file))
        images.append(os.path.join(data_dir,file))
    PathDF = pd.DataFrame({'images': images, 'masks': masks})
    train_df, valid_df = train_test_split(PathDF, random_state=42, test_size=0.2,shuffle=True)
    #valid_df, test_df = train_test_split(valid_df, random_state=42, test_size=0.2,shuffle=True)
    return train_df, valid_df#,test_df

class BasicDataset(Dataset):
    def __init__(self, path_df, data_dir, transform=None):
        self.path_df = path_df
        self.transform = transform
        self.data_dir = data_dir
    def __len__(self):
        return self.path_df.shape[0]  # df结构的shape[0]就是样本数，shape[1]可以理解为特征数

    def __getitem__(self, idx):
        base_path = os.path.join(self.data_dir, self.path_df.iloc[idx]['directory'])
        img_path = os.path.join(base_path, self.path_df.iloc[idx]['images'])
        mask_path = os.path.join(base_path, self.path_df.iloc[idx]['masks'])
        image=Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        a=np.array(mask)
        augmentation = self.transform(image=np.array(image), mask=np.array(mask))
        image = augmentation['image']
        mask = augmentation['mask']
        #mask=torch.div(mask,255)
        mask = torch.where((mask/ 255)> 0.5, 1, 0)






        #mask_onehot = F.one_hot(mask.long(), 2).permute(2,0,1).float()

        #mask=mask.unsequeeze(0)



        # fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        # ax[0].imshow(image.detach().numpy().transpose(1, 2, 0))
        # ax[0].set_title(img_path)
        # #ax[1].imshow(mask.detach().numpy().transpose(1, 2, 0))
        # ax[1].imshow(mask.detach().numpy())
        # ax[1].set_title('Mask')
        # plt.show()

        return image, mask

class MedicineDataset(Dataset):
    def __init__(self, path_df,mode):
        self.mode = mode
        self.path_df = path_df
        self.val_tranforms = A.Compose([
            A.Resize(256, 256),
            A.Normalize(),
            ToTensorV2()])
        self.train_transform =  A.Compose([
            A.RandomResizedCrop((256, 256), scale=(0.8, 1.0), p=1.0),
            #A.Resize(256, 256),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT,
                               value=0, p=0.5),
            A.HorizontalFlip(p=0.5),  # 垂直翻转
            A.VerticalFlip(p=0.5),  # 水平翻转
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.4),
            A.OneOf([
                A.ToGray(p=1),
                A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0,p=1),
                A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, value=0,p=1),
                A.CLAHE(p=1),
                A.HueSaturationValue(p=1),
                A.ChannelShuffle(p=1),
                A.GridDropout(p=1),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
                A.GaussianBlur(p=1)
                     ], p=0.4),  # 垂直翻转
            #A.GaussianBlur(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return self.path_df.shape[0]
        #return len(self.imgs_list)  # df结构的shape[0]就是样本数，shape[1]可以理解为特征数
    def __getitem__(self, idx):
        img_path = self.path_df.iloc[idx]['images']
        mask_path = self.path_df.iloc[idx]['masks']
        image=cv2.imread(img_path)
        mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        if self.mode == 'train':
            augmentation = self.train_transform(image=image, mask=mask)
            image = augmentation['image']
            mask = augmentation['mask']
            return image, mask.long()
        #mask = torch.where((mask/ 255)> 0.5, 1, 0)
        if self.mode == 'val':
            augmentation = self.val_tranforms(image=image, mask=mask)
            image = augmentation['image']
            mask = augmentation['mask']

            return image, mask.long()


class ThyroidDataset(Dataset):
    def __init__(self, path_df,transform=None):
        self.path_df = path_df
        self.transform = transform
    def __len__(self):
        return self.path_df.shape[0]
        #return len(self.imgs_list)  # df结构的shape[0]就是样本数，shape[1]可以理解为特征数
    def __getitem__(self, idx):
        img_path =  self.path_df.iloc[idx]['images']
        mask_path = self.path_df.iloc[idx]['masks']

        image=cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        masks = np.where(masks  > 0, 1, 0)
        augmentation = self.transform(image=image, mask=masks)
        image = augmentation['image']
        mask = augmentation['mask']
        return image, mask


class CVCDataset(Dataset):
    def __init__(self, path_df,transform=None):
        self.path_df = path_df
        self.transform = transform

    def __len__(self):
        return self.path_df.shape[0]

    def __getitem__(self, idx):
        base_path =  self.path_df.iloc[idx]['directory']
        img_path = os.path.join(base_path, self.path_df.iloc[idx]['images'])
        mask_path = os.path.join(base_path, self.path_df.iloc[idx]['masks'])
        image=Image.open(img_path).convert('RGB')
        masks=Image.open(mask_path).convert('L')

        augmentation = self.transform(image=np.array(image), mask=np.array(masks))
        image = augmentation['image']
        mask = augmentation['mask']

        mask = torch.where((mask/ 255)> 0.5, 1, 0)
        return image, mask.long()

class UBSIDataset(Dataset):
    def __init__(self, path_df,transform=None):
        self.path_df = path_df
        self.transform = transform
        # self.imgs_dir = os.path.join(data_dir+model,'images')
        # self.masks_dir = os.path.join(data_dir+model,'masks')
        # self.imgs_list = os.listdir(self.imgs_dir)
        # self.masks_list = os.listdir(self.masks_dir)
        # self.transform = transform

    def __len__(self):
        return self.path_df.shape[0]
        #return len(self.imgs_list)  # df结构的shape[0]就是样本数，shape[1]可以理解为特征数
    def __getitem__(self, idx):
        base_path =  self.path_df.iloc[idx]['directory']
        img_path = os.path.join(base_path, self.path_df.iloc[idx]['images'])
        mask_path = os.path.join(base_path, self.path_df.iloc[idx]['masks'])
        # img_path = os.path.join(self.imgs_dir, self.imgs_list[idx])
        # mask_path = os.path.join(self.masks_dir, self.masks_list[idx])
        image=cv2.imread(img_path)
        masks=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        # image=Image.open(img_path).convert('RGB')
        #mask = np.array(Image.open(mask_path))
        #transforms.ToTensor()
        augmentation = self.transform(image=image, mask=masks)
        image = augmentation['image']
        mask = augmentation['mask']
        #mask=torch.div(mask,255)
        mask = torch.where((mask/ 255)> 0.5, 1, 0)
        return image, mask

class My_VOC(VOCSegmentation):
    def __getitem__(self, idx):
        image=Image.open(self.images[idx]).convert('RGB')
        masks=Image.open(self.masks[idx])
        # image=Image.open(img_path).convert('RGB')
        #maska = np.array(Image.open(self.masks[idx]))
        #transforms.ToTensor()
        augmentation = self.transform(image=np.array(image), mask=np.array(masks))
        image = augmentation['image']
        mask = augmentation['mask'].long()
        #mask[mask > 20] = 0
        #mask=torch.div(mask,255)
        #mask = torch.where((mask/ 255)> 0.5, 1, 0)
        return image, mask

def calculate_sample_weights(data_frame, num_classes=2):
    diagnosis=data_frame['diagnosis'].values
    count = np.bincount(diagnosis, minlength=num_classes)
    total_frequency=np.sum(count)
    sample_weight = 1 / (np.log(1.02 + (count / total_frequency)))
    return sample_weight

def calculate_class_weights(dataloader, num_classes=2):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for image,target in tqdm_batch:
##这里是作者创建的一个dataloader，这里的sample['label']返回的是标签图像的lable mask
        y = target.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)##统计每幅图像中不同类别像素的个数
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))##这里是计算每个类别像素的权重
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    # classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')##生成权重文件
    # np.save(classes_weights_path, ret)##把各类别像素权重保存到一个文件中

    return ret

def addpicture(data_img):
    dirs, images, masks, diagnosis = [], [], [], []
    patical = []
    for root, folders, files in os.walk(data_img):  # 在目录树中游走,会打开子文件夹
        for file in files:
            if 'mask_1' in file:
                patical.append(file)
    for root, folders, files in os.walk(data_img):  # 在目录树中游走,会打开子文件夹
        for file in files:
            if file in patical:
                mask0 = cv2.imread(os.path.join(root, file.replace('mask_1', 'mask')),cv2.IMREAD_GRAYSCALE)
                mask1 = cv2.imread(os.path.join(root, file),cv2.IMREAD_GRAYSCALE)
                newmask=cv2.add(mask0,mask1)
                cv2.imwrite(os.path.join(root, file.replace('mask_1', 'mask')),newmask)
                os.remove(os.path.join(root, file))
                fig, ax = plt.subplots(1, 4, figsize=(8, 4))
                ax[0].imshow(mask0)
                ax[1].imshow(mask1)
                ax[2].imshow(newmask)
                plt.show()



if __name__ == "__main__":
    #path = os.path.abspath(os.path.dirname(__file__))
    # a=os.getcwd()
    #b=os.path.abspath('')
    #data_img = r"/home/gxmdjzx/.pycharm_helpers/pycharm/py_project/Dataset_BUSI_with_GT/"
    set_seed(10086)
    create_KvasirDataset_df()
    # data_img = r"/home/uax/SCY/seg/dataset/Thyroid_data/image"
    # set_seed(42)
    # train_df, valid_df = create_Thyroid_Dataset_df(data_img)
    # train_df.to_csv("train_Thyroid_seg.csv", index=False)
    # valid_df.to_csv("valid_Thyroid_seg.csv", index=False)
    #test_df.to_csv("test_Kvasir-SEG.csv", index=False)
    print('down')
    exit()



