import ConfigParser
import torchvision

from utils import *
from dataloader import *
 
if __name__ == '__main__':
	config = ConfigParser.ConfigParser()
	config.read("config.ini")
	dataset = config.get("argument", "dataset")
	# import pdb
	# pdb.set_trace()
	shuffle = config.get("argument", "shuffle")
	bsize = config.get("argument", "bsize")
	local_path =  config.get("argument", "path")
	hsize = config.get("argument","hidden")

	img_size = eval(config.get("argument", "size"), {}, {})

 	local_path = local_path+dataset
	if dataset=='cityscape':
		cst = citydataLoader(local_path,split='train',img_size=img_size,dataset=dataset)
		trainloader = data.DataLoader(cst,batch_size=bsize, shuffle=shuffle)
	elif dataset =='gta5':
		images_base = os.path.join(local_path,'images')
    	annotations_base = os.path.join(local_path,'labels')
    	labelIds = getgta5List(annotations_base,'.png',True)
    	imageIds = getgta5List(images_base,'.png',False)
    	train_labelIds,train_images,test_labelIds,test_images = splitData(labelIds,imageIds)
    	train_labelIds,train_images,val_labelIds,val_images = splitData(train_labelIds,train_images)
    	gta5 = gta5dataLoader(train_labelIds,train_images,img_size=img_size)
    	trainloader = data.DataLoader(gta5,batch_size=bsize,shuffle=shuffle)
    


	for i, data in enumerate(trainloader):
		
		