import ConfigParser
import torchvision
import matplotlib.pyplot as plt

from augmentations import *
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

 	local_path = local_path+"/"+dataset
	if dataset=='cityscape':
		print(dataset)
		augmentations = Compose([Scale(2048)])
		cst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
		trainloader = data.DataLoader(cst, batch_size=bsize, shuffle=shuffle,num_workers=0)
	elif dataset =='gta5':
		augmentations = Compose([Scale(2048)])	
		# gst = gtaLoader(local_path,is_transform=True,augmentations=augmentations)
		# trainloader = data.DataLoader(gst, batch_size=bsize, shuffle=shuffle,num_workers=0)

	
	
	for i, data in enumerate(trainloader):
		