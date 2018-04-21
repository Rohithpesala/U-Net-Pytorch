import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data

from utils import getGlobList
from augmentations import *

from PIL import Image

class dataLoader(data.Dataset):

    def __init__(self, root,img_size=(512, 1024),dataset=None):
        self.root = root
        self.split = split
        self.img_size = img_size
        if dataset =='cityscape':
        	self.images_base = os.path.join(self.root,'gtFine',self.split)
        	self.annotations_base = os.path.join(self.root,'gtFine', self.split)
        if dataset == 'gta5':
        	self.images_base = os.path.join(self.root,'images')
        	self.annotations_base = os.path.join(self.root,'labels')
        
        self.labelIds = getGlobList(self.annotations_base,'.png',True)
        self.imageIds = getGlobList(self.images_base,'.png',False)
        print("Found %d %s images" % (len(self.labelIds), split))

    def __len__(self):
        return len(self.labelIds)

    def __getitem__(self, index):
        img_path = self.imageIds[index].rstrip()

        lbl_path = self.labelIds[index]

        img = Image.open(img_path)
        img = img.resize(self.img_size, Image.NEAREST)
        img = np.array(img,dtype=np.uint8)

        lbl = Image.open(lbl_path)
        lbl = lbl.resize(self.img_size,Image.NEAREST)
        lbl = np.array(lbl,dtype=np.uint8)
        
        return img, lbl  
