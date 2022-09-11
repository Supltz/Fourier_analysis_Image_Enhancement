from torch.utils.data.dataset import Dataset
import cv2
import pickle
import numpy as np
import torch
from glob import glob
import os
from torchvision import transforms

class Cifar10(Dataset):
    def __init__(self, data_path, mode='train'):
        self.data_path = data_path

        self.mode = mode
        self.image = []
        with open(self.data_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            for i in range(10000):
                image = np.reshape(dict[b'data'][i,:],[3,32,32])
                img = np.moveaxis(image,0,-1)
                img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                self.image.append(img_gray)

    def __getitem__(self, index):
      img_tensor = torch.Tensor(self.image[index])
      img_tensor = img_tensor.unsqueeze(dim=0)
      return img_tensor

    def __len__(self):
        return len(self.image)

class UWdataset(Dataset):
    def __init__(self, data_path, mode='train'):
        self.data_path = data_path

        self.mode = mode
        self.image = glob(os.path.join(self.data_path, '*.png'))
        self.transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((270, 360)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = cv2.imread(self.image[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = self.transform1(img)
        return img

    def __len__(self):
        return len(self.image)

class UIEB(Dataset):
    def __init__(self, data_path,gt_path,mode):
        self.data_path = data_path
        self.gt_path = gt_path
        self.mode = mode
        self.image = glob(os.path.join(self.data_path, '*.png'))
        self.groundtruth = glob(os.path.join(self.gt_path, '*.png'))
        if self.mode == "train":
            self.image = self.image[0:600]
            self.groundtruth = self.groundtruth[0:600]
        elif self.mode == "val":
            self.image = self.image[600:664]
            self.groundtruth = self.groundtruth[600:664]
        elif self.mode == "test":
            self.image = self.image[800:827]
            self.groundtruth = self.groundtruth[800:827]

        self.transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((270, 360)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = cv2.imread(self.image[index])
        img = self.transform1(img)
        gt = cv2.imread(self.groundtruth[index])
        gt = self.transform1(gt)
        return img,gt

    def __len__(self):
        return len(self.image)

class NYUUWDataset(Dataset):
    def __init__(self, data_path, label_path, img_format='png', size=30000, mode='train', train_start=0, val_start=30000, test_start=33000):
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.size = size
        self.train_start = train_start
        self.test_start = test_start
        self.val_start = val_start

        self.uw_images = glob(os.path.join(self.data_path, '*.' + img_format))  # glob get each image path in the data file, return a list([])

        if self.mode == 'train':
            self.uw_images = self.uw_images[self.train_start:self.train_start+self.size]
        elif self.mode == 'val':
            self.uw_images = self.uw_images[self.val_start:self.val_start+self.size]

        self.cl_images = []  # label path + image number + .png

        for img in self.uw_images:
            self.cl_images.append(os.path.join(self.label_path, os.path.basename(img).split('_')[0]  + '.' + img_format))

        for uw_img, cl_img in zip(self.uw_images, self.cl_images):
            assert os.path.basename(uw_img).split('_')[0] == os.path.basename(cl_img).split('.')[0], ("Files not in sync.")

        self.transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((270, 360)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor()
            ])
    def __getitem__(self, index):

            uw_img = cv2.imread(self.uw_images[index])
            cl_img = cv2.imread(self.cl_images[index])
            uw_img = cv2.cvtColor(uw_img, cv2.COLOR_BGR2RGB)
            cl_img = cv2.cvtColor(cl_img, cv2.COLOR_BGR2RGB)

            name = os.path.basename(self.uw_images[index])[:-4]
            uw_img = self.transform1(uw_img)
            cl_img = self.transform1(cl_img)

            return uw_img,cl_img,name

    def __len__(self):
        return self.size

class Facade(Dataset):
    def __init__(self, data_path,gt_path,mode):
        self.data_path = data_path
        self.gt_path = gt_path
        self.mode = mode
        self.image = glob(os.path.join(self.data_path, '*.png'))
        self.groundtruth = [i.replace(".png",".jpg") for i in self.image]
        if self.mode == "train":
            self.image = self.image[0:500]
            self.groundtruth = self.groundtruth[0:500]
        elif self.mode == "val":
            self.image = self.image[500:564]
            self.groundtruth = self.groundtruth[500:564]
        elif self.mode == "test":
            self.image = self.image[564:600]
            self.groundtruth = self.groundtruth[564:600]

        self.transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((270, 360)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = cv2.imread(self.image[index])
        img = self.transform1(img)
        gt = cv2.imread(self.groundtruth[index])
        gt = self.transform1(gt)
        return img,gt

    def __len__(self):
        return len(self.image)

class Shoes(Dataset):
    def __init__(self, data_path,gt_path,mode):
        self.data_path = data_path
        self.gt_path = gt_path
        self.mode = mode
        self.image = glob(os.path.join(self.data_path, '*.jpg'))
        if self.mode == "train":
            self.image = self.image[0:600]
        elif self.mode == "val":
            self.image = self.image[600:664]
        elif self.mode == "test":
            self.image = self.image[664:700]

        self.transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_all = cv2.imread(self.image[index])
        img_all = self.transform1(img_all)
        img = img_all[:,:,0:256]
        gt = img_all[:,:,256:512]
        return img,gt

    def __len__(self):
        return len(self.image)