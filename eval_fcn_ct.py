import logging
import os
import os.path as osp
from collections import deque
import sys
sys.path.append('D:/cycada_2/cycada_2')
import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from PIL import Image
import time
from torch.autograd import Variable
from datetime import datetime
import pytz
import re
import medpy.metric.binary as mmb
# from cycada.data.mmwhsdata import Importmmwhs
# from cycada.data.crossmadata import Importcrossmoda
from cycada.data.data_loader import get_fcn_dataset as get_dataset
from cycada.models import get_model
from cycada.models.models import models
from cycada.transforms import augment_collate
from cycada.util import config_logging
from cycada.util import to_tensor_raw
from cycada.util import checkpoint_save
from cycada.tools.util import make_variable
#from cycada.util import DiceLoss
from cycada.util import step_lr
from matplotlib import pyplot as plt
import pdb
import seaborn as sns

BASE_FID = 'D:/mmwhs_sifa/mmwhs_sifa/npz' # folder path of test files
TESTFILE_FID = 'test_datalist.txt' # path of the .txt file storing the test filenames
TEST_MODALITY = 'CT'
data_size = [256, 256, 1]
label_size = [256, 256, 1]
datadir = ''

def get_cur_time():
    return datetime.strftime(datetime.now(pytz.timezone('Asia/Singapore')), '%Y-%m-%d_%H-%M-%S')

def read_lists(base_fd, fid):
    """read test file list """

    with open(fid, 'r') as fd:
        _list = fd.readlines()

    my_list = []
    for _item in _list:
        my_list.append(base_fd + '/' + _item.split()[0])
    return my_list
 
@click.command()
@click.option('--path', default= 'D:/cycada_2/cycada_2/results/fake_ct_to_ct/fcn8s/211111-0137/bestnet-iter2300.pth')
@click.option('--dataset', default='')
@click.option('--datadir', default=datadir)
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--gpu', default='0')
@click.option('--num_cls', default=5)

def main(path, dataset, datadir, model, gpu, num_cls):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = get_model(model, num_cls=num_cls)
    net.load_state_dict(torch.load(path))
    net.to(device)
    net.eval()

    test_fid = BASE_FID + '/' + TESTFILE_FID
    test_list = read_lists(BASE_FID,test_fid)
    
    dice_list = []
    assd_list = []   
    for idx_file, fid in enumerate(test_list):
        _npz_dict = np.load(fid)
        data = _npz_dict['arr_0']
        label = _npz_dict['arr_1']
        predict = np.zeros(label.shape)

        data = (data - data.mean())/data.std()
        
        if True:
            data = np.flip(data, axis=0)
            data = np.flip(data, axis=1)
            label = np.flip(label, axis=0)
            label = np.flip(label, axis=1)
        # if TEST_MODALITY == 'CT':
        #     data = np.rot90(data,3, axes=(1,2))
        #     data = np.flip(data, axis=1)
        #     label = np.rot90(label,3, axes=(1,2))
        #     label = np.flip(label, axis=1)
        with torch.no_grad():        
            for i in range(data.shape[0]):
                batch_img = np.expand_dims(data[..., i].copy(), 0)
                batch_img = torch.tensor(batch_img[np.newaxis,...]).type(torch.FloatTensor).to(device)
                #print(batch_img.shape)

                output = net(batch_img)
                #print(type(output))

                output = torch.softmax(output, dim = 1)
                tmp_pred = torch.argmax(output, dim = 1).cpu().numpy()
                # print(output.shape,tmp_pred.shape)
                predict[:,:,i] = tmp_pred[0,:,:]

                # print(fid)
                
                # Visualisation
                if ( i >= 110 and i <= 120):
                    colors = ["black", "mediumblue", "tab:pink", "mediumseagreen", "orange"]

                    tmp_pred2 = tmp_pred[0,:,:]
                    ax = sns.heatmap(tmp_pred2,cmap=colors, cbar=False)
                    plt.gca().set_aspect('equal')
                    plt.axis('off')
                    plt.show()

                    # plt.imshow(label[:,:,i],cmap=colors)
                    # plt.show()

                    ax = sns.heatmap(label[:,:,i],cmap=colors, cbar=False)
                    plt.gca().set_aspect('equal')
                    plt.axis('off')
                    plt.show()


                    batch_img_store = batch_img.cpu().data.numpy()
                    batch_img_store = batch_img_store[0,0,:,:]
                    plt.imshow(batch_img_store, cmap='gray')
                    plt.axis('off')
                    plt.show()

        #pdb.set_trace()
        # print(predict.shape, label.shape)
        for c in range(1, num_cls):
            pred_test_data_tr = predict.copy()
            pred_test_data_tr[pred_test_data_tr != c] = 0
            print(np.unique(pred_test_data_tr))

            pred_gt_data_tr = label.copy()
            pred_gt_data_tr[pred_gt_data_tr != c] = 0

            dice_list.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))
            try:
                assd_list.append(mmb.assd(pred_test_data_tr, pred_gt_data_tr))
            except:
                assd_list.append(np.nan)

    print(dice_list)
    print(assd_list)
    dice_arr = 100 * np.reshape(dice_list, [4, -1]).transpose()

    dice_mean = np.mean(dice_arr, axis=1)
    dice_std = np.std(dice_arr, axis=1)

    print('Dice:')
    print('AA :%.1f(%.1f)' % (dice_mean[3], dice_std[3]))
    print('LAC:%.1f(%.1f)' % (dice_mean[1], dice_std[1]))
    print('LVC:%.1f(%.1f)' % (dice_mean[2], dice_std[2]))
    print('Myo:%.1f(%.1f)' % (dice_mean[0], dice_std[0]))
    print('Mean:%.1f' % np.mean(dice_mean))

    assd_arr = np.reshape(assd_list, [4, -1]).transpose()

    assd_mean = np.mean(assd_arr, axis=1)
    assd_std = np.std(assd_arr, axis=1)

    print('ASSD:')
    print('AA :%.1f(%.1f)' % (assd_mean[3], assd_std[3]))
    print('LAC:%.1f(%.1f)' % (assd_mean[1], assd_std[1]))
    print('LVC:%.1f(%.1f)' % (assd_mean[2], assd_std[2]))
    print('Myo:%.1f(%.1f)' % (assd_mean[0], assd_std[0]))
    print('Mean:%.1f' % np.mean(assd_mean))

        
if __name__ == '__main__':
    main()
