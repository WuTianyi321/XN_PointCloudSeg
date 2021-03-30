import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
import open3d as o3d


class XN_PointCloudDataset(data.Dataset):
    def __init__(self,root,split='train',npoints=2500):
        self.root = root
        self.npoints=npoints
        self.num_seg_classes=2
        test1='1608347279931'
        test2='1608349012229'
        if split=='train':
            self.dirs=os.listdir(self.root)
            self.dirs.remove(test1)
            self.dirs.remove(test2)
        else:
            self.dirs=[test1,test2]


    def __getitem__(self,index):
        dirname = self.root + self.dirs[index]
        point_set = np.asarray(o3d.io.read_point_cloud(dirname +'\\PointCloudCapture.pcd').points).astype(np.int64)
        seg_classes = np.loadtxt(dirname + '\\label.txt').astype(np.float32)
        choice = np.random.choice(len(seg_classes), self.npoints, replace=True)
        point_set=point_set[choice,:]
        seg=torch.from_numpy(seg_classes[choice])
        point_set = torch.from_numpy(point_set)
        return point_set,seg
    
    def __len__(self):
        return len(self.dirs)