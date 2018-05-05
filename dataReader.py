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
     	# gta5_train = dataLoader(train_labelIds,train_images,img_size=img_size)
     	# gta5_val = dataLoader(val_labelIds,val_images,img_size=img_size)
     	# gta5_test = dataLoader(test_labelIds,test_images,img_size=img_size)
     	# trainLoaderDict['train'] = data.DataLoader(gta5_train,batch_size=args.batch_size,shuffle=args.shuffle)
     	# trainLoaderDict['test'] = data.DataLoader(gta5_test,batch_size=args.batch_size,shuffle=args.shuffle)
     	# trainLoaderDict['validation'] = data.DataLoader(gta5_val,batch_size=args.batch_size,shuffle=args.shuffle)
     if(args.data_type=='cityscape' or args.data_type == 'joint'):
          paths = os.path.join(path,'cityscape')
          img_dir_name = 'leftImg8bit_trainvaltest'
          lbl_suffix = '_gtFine_labelIds.png'
          train_labelIds = train_labelIds + getList(os.path.join(paths, 'gtFine_trainvaltest', 'gtFine', 'train'),lbl_suffix,True)
          val_labelIds = val_labelIds + getList(os.path.join(paths, 'gtFine_trainvaltest', 'gtFine', 'val'),lbl_suffix,True)
          test_labelIds = test_labelIds + getList(os.path.join(paths, 'gtFine_trainvaltest', 'gtFine', 'test'),lbl_suffix,True)
          train_images = train_images + getList(os.path.join(paths,img_dir_name,'leftImg8bit','train'),'.png',False)
          val_images = val_images + getList(os.path.join(paths,img_dir_name,'leftImg8bit','val'),'.png',False)
          test_images= test_images+ getList(os.path.join(paths,img_dir_name,'leftImg8bit','test'),'.png',False)
          # cityscape_train = dataLoader(labelIds_train,img_train,img_size=img_size)
          # cityscape_test = dataLoader(labelIds_test,img_test,img_size=img_size)
          # cityscape_val = dataLoader(labelIds_val,img_val,img_size=img_size)
          # trainLoaderDict['train'] = data.DataLoader(cityscape_train,batch_size=args.batch_size,shuffle=args.shuffle)
          # trainLoaderDict['test'] = data.DataLoader(cityscape_test,batch_size=args.batch_size,shuffle=args.shuffle)
          # trainLoaderDict['validation'] = data.DataLoader(cityscape_val,batch_size=args.batch_size,shuffle=args.shuffle)

     train = dataLoader(train_labelIds,train_images,img_size=img_size)
     val = dataLoader(val_labelIds,val_images,img_size=img_size)
     test = dataLoader(test_labelIds,test_images,img_size=img_size)
     trainLoaderDict['train'] = data.DataLoader(train,batch_size=args.batch_size,shuffle=args.shuffle)
     trainLoaderDict['test'] = data.DataLoader(test,batch_size=args.batch_size,shuffle=args.shuffle)
     trainLoaderDict['validation'] = data.DataLoader(val,batch_size=args.batch_size,shuffle=args.shuffle)
     return trainLoaderDict