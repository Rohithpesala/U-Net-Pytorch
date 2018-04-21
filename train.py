import ConfigParser
import torchvision

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
	img_size = config.get("argument","size")
 	local_path = local_path+"/"+dataset
	if dataset=='cityscape':
		cst = dataLoader(local_path,img_size=img_size,dataset=dataset)
		trainloader = data.DataLoader(cst, batch_size=bsize, shuffle=shuffle)
	elif dataset =='gta5':
		gta5 = dataLoader(local_path,img_size=img_size,dataset=dataset)
		print("done")
		trainloader = data.DataLoader(gta5,batch_size=bsize,shuffle=shuffle)
		
	
	for i, data in enumerate(trainloader):
		import pdb
		pdb.set_trace()
		