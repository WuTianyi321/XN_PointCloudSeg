import torch
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm

root = 'D:\\Users\\WuTianyi\\OneDrive - wutianyidev\\NEU\\课程\\MATH7243 Machine Learning\\大作业\\Cloud Sample Data\\'

dirs=os.listdir(root)
for dir_name in dirs:
    curr_dir = os.path.join(root, dir_name)
    pcd_face = o3d.io.read_point_cloud(curr_dir+'\\face_segment.pcd')
    pcd_all = o3d.io.read_point_cloud(curr_dir+'\\PointCloudCapture.pcd')
    face_seg=torch.tensor(pcd_face.points)
    all_data=torch.tensor(pcd_all.points)
    label=torch.zeros(len(all_data))
    for i in tqdm(face_seg):
        label[np.argwhere(
            torch.sum((i==all_data)
            ,dim=1)==3)[0][0]]=1
    np.savetxt(curr_dir+'\\label.txt',label,fmt='%d')