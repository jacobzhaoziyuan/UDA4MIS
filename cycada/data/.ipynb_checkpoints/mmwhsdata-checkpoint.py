from torch.utils.data import Dataset
import os
import os.path as osp
import numpy as np
import re
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from torchvision import transforms

totensor = transforms.ToTensor()
class Importmmwhs(Dataset):

    def __init__(self, phases=['train'], data='mr', data_dir='/home/ziyuan/UDA/data/mmwhs_sifa/'):
        self.phases = phases
        self.data = data
        self.image_lst = []
        self.label_lst = []
        for phase in phases:
            data_path = os.path.join(data_dir, data, data+'_'+phase)
            image_lst_ = [osp.join(data_path,f) for f in os.listdir(data_path) if 'label' not in f]
            label_lst_ = [re.sub('.npy','_label.npy', f) for f in image_lst_]
            self.image_lst += image_lst_
            self.label_lst += label_lst_

        print("total {} samples".format(len(self.image_lst)))

    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, idx):
        img_path = self.image_lst[idx]
        label_path = self.label_lst[idx]
        img = np.load(img_path)[...,0]
        label = np.load(label_path)
        # print('shape',img.shape,label.shape)
        sample = img[np.newaxis,...],  label.astype(np.uint8)
        return sample

if __name__ == '__main__':
    set = Importmmwhs(phases=['val'],data = 'ct')
    img, label = set[0]
    print(img.shape,label.shape)
