from torch.utils.data import Dataset
import os
import os.path as osp
import numpy as np
import re
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from torchvision import transforms
import random

totensor = transforms.ToTensor()

class Importmmwhs_adda(Dataset):

    def __init__(self,phases=['train'], data_dir='/home/ziyuan/UDA/data/mmwhs_sifa/'):
        self.phases = phases
        self.mr_image_lst = []
        self.mr_label_lst = []
        self.ct_image_lst = []
        self.ct_label_lst = []
        for phase in phases:
            mr_path = osp.join(data_dir, 'mr', 'mr_'+phase) 
            mr_img_lst =  [osp.join(mr_path,f) for f in os.listdir(mr_path) if 'label' not in f]
            mr_label_lst =[re.sub('.npy','_label.npy', f) for f in mr_img_lst]
            self.mr_image_lst += mr_img_lst
            self.mr_label_lst += mr_label_lst

            ct_path = osp.join(data_dir, 'ct', 'ct_'+phase) 
            ct_img_lst =  [osp.join(ct_path,f) for f in os.listdir(ct_path) if 'label' not in f]
            ct_label_lst =[re.sub('.npy','_label.npy', f) for f in ct_img_lst]
            # print(ct_label_lst[:10])
            self.ct_image_lst += ct_img_lst
            self.ct_label_lst += ct_label_lst

        self.mr_size = len(self.mr_image_lst)
        self.ct_size = len(self.ct_image_lst)
        
        print("total {}, {}samples".format(self.mr_size, self.ct_size))

    def __len__(self):
        return max(len(self.mr_image_lst), len(self.ct_image_lst))

    def __getitem__(self, idx):
        mr_img_path = self.mr_image_lst[idx % self.mr_size] #  mr_1001_11_fake_B.npy
        mr_label_path = self.mr_label_lst[idx % self.mr_size] #  mr_1001_11_fake_B.npy
        mr_img = np.load(mr_img_path)[...,0]
        mr_label = np.load(mr_label_path)

        ind_ct = random.randint(0, self.ct_size - 1)
        ct_img_path = self.ct_image_lst[ind_ct]
        ct_label_path = self.ct_label_lst[ind_ct]
        ct_img = np.load(ct_img_path)[...,0]
        ct_label = np.load(ct_label_path)
        # print(totensor(mr_img).float().shape, totensor(ct_img[np.newaxis,:]).float().shape, mr_label.astype(np.uint8).shape, ct_label.astype(np.uint8).shape)

        return (mr_img[np.newaxis,:], ct_img[np.newaxis,:], 
                mr_label.astype(np.uint8), ct_label.astype(np.uint8))
    
if __name__ == '__main__':
    set = Importmmwhs_adda(phases=['train','val'])
    mr_img,  ct_img,mr_label,ct_label = set[0]
    print(mr_img.shape,mr_label.shape,ct_img.shape,ct_label.shape)
