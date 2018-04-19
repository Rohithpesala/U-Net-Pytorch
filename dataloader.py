import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data

from utils import getGlobList
from augmentations import *


class cityscapesLoader(data.Dataset):

    def __init__(self, root, split="train", is_transform=False, 
                 img_size=(512, 1024), augmentations=None):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 19
        self.img_size = img_size
        # self.mean = np.array([73.15835921, 82.90891754, 72.39239876])
        self.files = {}

        self.images_base = os.path.join(self.root,'gtFine',self.split)
        self.annotations_base = os.path.join(self.root,'gtFine', self.split)

        # self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.png')
        
        self.labelIds = getGlobList(self.annotations_base,'.png',True)
        self.imageIds = getGlobList(self.images_base,'.png',False)

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',\
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',\
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19))) 
        print("Found %d %s images" % (len(self.labelIds), split))

    def __len__(self):
        return len(self.labelIds)

    def __getitem__(self, index):
        img_path = self.imageIds[index].rstrip()

        lbl_path = self.labelIds[index]
        print(img_path)
        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        print(lbl_path)
        lbl = m.imread(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)
        
        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1)
        
        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        lbl = lbl.astype(int)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


    def encode_segmap(self, mask):
        for _voidc in self.void_classes:
            mask[mask==_voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask==_validc] = self.class_map[_validc]
        return mask



