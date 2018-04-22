from collections import OrderedDict
import os
import numpy as np
from sklearn.model_selection import train_test_split
def getGlobList(rootdir='.', suffix='',label=False):
    if label == True:
    	return [os.path.join(looproot, filename)
        	for looproot, _, filenames in os.walk(rootdir)
        	for filename in filenames if (filename.endswith(suffix) and filename.split("_")[-1]=='labelIds.png' )]
    else:
    	return [os.path.join(looproot, filename)
        	for looproot, _, filenames in os.walk(rootdir)
        	for filename in filenames if (filename.endswith(suffix) and filename.split("_")[-1]=='color.png' )]

def getgta5List(rootdir='.',suffix='',label=False):
	if label == True:
		return [os.path.join(looproot, filename)
			for looproot, _, filenames in os.walk(rootdir)
			for filename in filenames if filename.endswith(suffix)]
	else:
		return [os.path.join(looproot, filename)
        	for looproot, _, filenames in os.walk(rootdir)
        	for filename in filenames if filename.endswith(suffix)]

def splitData(labelIds,images):
	train_images,test_images,train_labels,test_labels = train_test_split(images,labelIds,test_size=0.2,random_state=42)
   	return train_labels,train_images,test_labels,test_images