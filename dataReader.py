import ConfigParser
import torchvision

from utils import *
from loader import *
 
def dataReader(args):
     path=args.data_dir
     img_size = (args.image_height,args.image_width)
     trainLoaderDict = {}
     train_labelIds = []
     train_images = []
     test_labelIds = []
     test_images = []
     val_labelIds = []
     val_images = []
     if(args.data_type=='gta5' or args.data_type == 'joint'):
          paths = os.path.join(path,'gta5')
          images_base = os.path.join(paths,'images')
          annotations_base = os.path.join(paths,'labels')
          labelIds = getList(annotations_base,'.png',True)
          imageIds = getList(images_base,'.png',False)
          train_labelIds,train_images,test_labelIds,test_images = splitData(labelIds,imageIds)
          train_labelIds,train_images,val_labelIds,val_images = splitData(train_labelIds,train_images)
     if(args.data_type=='cityscape' or args.data_type == 'joint'):
          paths = os.path.join(path,'cityscape')
          lbl_suffix = '_gtFine_labelIds.png'
          lable,img = getCityList(os.path.join(paths, 'images', 'train'),os.path.join(paths, 'labels', 'train'),lbl_suffix)
          train_labelIds = train_labelIds + lable
          train_images = train_images + img
          lable,img = getCityList(os.path.join(paths, 'images', 'test'),os.path.join(paths, 'labels', 'test'),lbl_suffix)
          test_labelIds = test_labelIds + lable
          test_images = test_images + img
          lable,img = getCityList(os.path.join(paths, 'images', 'val'),os.path.join(paths, 'labels', 'val'),lbl_suffix)
          val_labelIds = val_labelIds + lable
          val_images = val_images + img
           

     train = dataLoader(train_labelIds,train_images,img_size=img_size)
     val = dataLoader(val_labelIds,val_images,img_size=img_size)
     test = dataLoader(test_labelIds,test_images,img_size=img_size)
     trainLoaderDict['train'] = data.DataLoader(train,batch_size=args.batch_size,shuffle=args.shuffle)
     trainLoaderDict['test'] = data.DataLoader(test,batch_size=args.batch_size,shuffle=args.shuffle)
     trainLoaderDict['validation'] = data.DataLoader(val,batch_size=args.batch_size,shuffle=args.shuffle)
     return trainLoaderDict