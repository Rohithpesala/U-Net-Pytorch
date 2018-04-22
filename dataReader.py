import ConfigParser
import torchvision

from utils import *
from loader import *
 
# if __name__ == '__main__':
# 	config = ConfigParser.ConfigParser()
# 	config.read("config.ini")
# 	dataset = config.get("argument", "dataset")
# 	# import pdb
# 	# pdb.set_trace()
# 	shuffle = config.get("argument", "shuffle")
# 	bsize = config.get("argument", "bsize")
# 	local_path =  config.get("argument", "path")
# 	hsize = config.get("argument","hidden")

# 	img_size = eval(config.get("argument", "size"), {}, {})

#  	local_path = "/mnt/c/U-Net-Pytorch/data/gta5-dummy"
#  	bsize = 4
# 	if dataset=='cityscape':
# 		cst = citydataLoader(local_path,split='train',img_size=img_size,dataset=dataset)
# 		trainloader = data.DataLoader(cst,batch_size=bsize, shuffle=shuffle)
# 	elif dataset =='gta5':
# 		images_base = os.path.join(local_path,'images')
#     	annotations_base = os.path.join(local_path,'labels')
#     	import pdb
#     	pdb.set_trace()
#     	labelIds = getgta5List(annotations_base,'.png',True)
#     	imageIds = getgta5List(images_base,'.png',False)
#     	train_labelIds,train_images,test_labelIds,test_images = splitData(labelIds,imageIds)
#     	train_labelIds,train_images,val_labelIds,val_images = splitData(train_labelIds,train_images)
#     	gta5 = gta5dataLoader(train_labelIds,train_images,img_size=img_size)
#     	trainloader = data.DataLoader(gta5,batch_size=bsize,shuffle=shuffle)
    


# 	for i, data in enumerate(trainloader):
# 		print(i,data)
# 		break
		


def dataReader(args):
     path=args.data_dir
     img_size = (args.image_height,args.image_width)
     trainLoaderDict = {}
     if(args.data_type=='gta5'):
     	images_base = os.path.join(path,'images')
     	annotations_base = os.path.join(path,'labels')
     	labelIds = getgta5List(annotations_base,'.png',True)
     	imageIds = getgta5List(images_base,'.png',False)
     	train_labelIds,train_images,test_labelIds,test_images = splitData(labelIds,imageIds)
     	train_labelIds,train_images,val_labelIds,val_images = splitData(train_labelIds,train_images)
     	gta5_train = gta5dataLoader(train_labelIds,train_images,img_size=img_size)
     	gta5_val = gta5dataLoader(val_labelIds,val_images,img_size=img_size)
     	gta5_test = gta5dataLoader(test_labelIds,test_images,img_size=img_size)
     	trainLoaderDict['train'] = data.DataLoader(gta5_train,batch_size=args.batch_size,shuffle=args.shuffle)
     	trainLoaderDict['test'] = data.DataLoader(gta5_test,batch_size=args.batch_size,shuffle=args.shuffle)
     	trainLoaderDict['validation'] = data.DataLoader(gta5_val,batch_size=args.batch_size,shuffle=args.shuffle)
     return trainLoaderDict