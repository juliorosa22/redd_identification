import os,sys
import math
import numpy as np
import h5py as h5
import csv 


curr_work_dir=os.getcwd()

def saveH5File(file_path,labels,data):
    if len(labels) == len(data):
        f = h5.File(file_path,'w')
        for i in range(len(labels)):
            f_dset=f.create_dataset(labels[i],data[i].shape,dtype=data[i].dtype)
            f_dset[:]=data[i][:]
        f.close()        
    else:
        print('Error, labels and data must be same len.')

def readH5File(file_path):
    f=h5.File(file_path,'r')
    keys = list(f)
    data = [ f[k][:] for k in keys ]
    return (keys,data)