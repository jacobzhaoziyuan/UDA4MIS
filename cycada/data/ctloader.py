from torch.utils.data import Dataset
import os
import os.path as osp
import numpy as np
import re
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from torchvision import transforms

class CTDataSet(data.Dataset):
    def __init__(self, root, list_path=None, max_iters=None, crop_size=(512, 512), mean=0, scale=True, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = osp.join(root, set+'.txt')
        self.crop_size = crop_size
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.set = set

        self.image_lst = []
        self.volume_lst = [1003,1008,1014,1019]
        for i in self.volume_lst:
            image_lst = [f for f in self.img_ids if f.startswith('ct_test/ct_'+str(i))]
            image_lst.sort()
            # print(image_lst[:10])
            self.image_lst.append(image_lst)


    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, index):

        image_lst = self.image_lst[index]
        volume_id = self.volume_lst[index]
        volume_img = np.zeros((len(image_lst),1,self.crop_size[0],self.crop_size[1]))
        volume_label = np.zeros((len(image_lst),self.crop_size[0],self.crop_size[1]))
        for i in range(len(image_lst)):
            name = 'ct_test/ct_%s_%s.npy'%(volume_id, i)
            img_file = os.path.join(self.root, "%s" % ( name))
            label_file = os.path.join(self.root, re.sub('.npy','_label.npy',name))

            image = np.load(img_file)
            label = np.load( label_file)

            # resize
            image_resize = cv2.resize(image, self.crop_size, interpolation = cv2.INTER_CUBIC)
            image_resize -= self.mean
            # print(image_resize.max(), image_resize.min())
            label_resize = cv2.resize(label, self.crop_size, interpolation = cv2.INTER_NEAREST)
            volume_img[i,...] = image_resize.copy()[np.newaxis,...]
            volume_label[i,...] = label_resize.copy()

        return volume_img,  volume_label.astype(np.uint8)

if __name__ == '__main__':
    set = Importmmwhs(phases=['val'],data = 'ct')
    img, label = set[0]
    print(img.shape,label.shape)
