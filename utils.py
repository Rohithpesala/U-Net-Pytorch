from collections import OrderedDict
import os
import numpy as np



def getGlobList(rootdir='.', suffix='',label=False):
    if label == True:
    	return [os.path.join(looproot, filename)
        	for looproot, _, filenames in os.walk(rootdir)
        	for filename in filenames if (filename.endswith(suffix) and filename.split("_")[-1]=='labelIds.png' )]
    else:
    	return [os.path.join(looproot, filename)
        	for looproot, _, filenames in os.walk(rootdir)
        	for filename in filenames if (filename.endswith(suffix) and filename.split("_")[-1]=='color.png' )]