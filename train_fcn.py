import logging
from matplotlib import pyplot as plt
import os
import os.path as osp
from collections import deque
import sys

from torch.utils import data
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
import medpy.metric.binary as mmb

from cycada.data.mmwhsdata import Importmmwhs
from cycada.data.data_loader import get_fcn_dataset as get_dataset
from cycada.models import get_model
from cycada.models.models import models
from cycada.transforms import augment_collate
from cycada.util import config_logging
from cycada.util import to_tensor_raw
from cycada.tools.util import make_variable

#BASE_FID = 'D:/mmwhs_sifa/mmwhs_sifa/new_transformed_data/fake_mr/fake_mr_test_npz' # folder path of test files
BASE_FID = 'D:/mmwhs_sifa/mmwhs_sifa/npz' # folder path of test files
TESTFILE_FID = 'test_datalist.txt' # path of the .txt file storing the test filenames
TEST_MODALITY = 'CT'
data_size = [256, 256, 1]
label_size = [256, 256, 1]

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

def to_tensor_raw(im):
    return torch.from_numpy(np.array(im, np.int64, copy=False))


def roundrobin_infinite(*loaders):
    if not loaders:
        return
    iters = [iter(loader) for loader in loaders]
    while True:
        for i in range(len(iters)):
            it = iters[i]
            try:
                yield next(it)
            except StopIteration:
                iters[i] = iter(loaders[i])
                yield next(iters[i])

def supervised_loss(score, label, weights=None):
    loss_fn_ = torch.nn.NLLLoss2d(weight=weights, size_average=True, 
            ignore_index=255)
    loss = loss_fn_(F.log_softmax(score,dim=1), label)
    return loss
 

@click.command()
@click.argument('output')
@click.option('--dataset', required=True, multiple=True)
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--batch_size', '-b', default=1)
@click.option('--lr', '-l', default=0.001)
@click.option('--step', type=int)
@click.option('--epochs', '-e', default=100, help='max epoch to run')
@click.option('--momentum', '-m', default=0.9)
@click.option('--snapshot', '-s', default=5000)
@click.option('--downscale', type=int)
@click.option('--augmentation/--no-augmentation', default=False)
@click.option('--fyu/--torch', default=False)
@click.option('--crop_size', default=720)
@click.option('--weights', type=click.Path(exists=True))
@click.option('--model', default='drn26', type=click.Choice(models.keys()))
@click.option('--num_cls', default=5, type=int)
@click.option('--gpu', default='0')
def main(output, dataset, datadir, batch_size, lr, step, epochs, 
        momentum, snapshot, downscale, augmentation, fyu, crop_size, 
        weights, model, gpu, num_cls):
    if weights is not None:
        raise RuntimeError("weights don't work because eric is bad at coding")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config_logging()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_fid = BASE_FID + '/' + TESTFILE_FID
    test_list = read_lists(BASE_FID,test_fid)
    
    logdir = 'runs/{:s}/{:s}'.format(model, '-'.join(dataset))
    output = osp.join(output, time.strftime('%y%m%d-%I%M', time.localtime()))
    # print(type(output))
    # print(output)
    if not os.path.exists(output):
        os.mkdir(output)
    writer = SummaryWriter(log_dir=logdir)
    net = get_model(model, num_cls=num_cls)
    net.cuda()
    # transform = []
    # target_transform = []
    # if downscale is not None:
    #     transform.append(torchvision.transforms.Scale(1024 // downscale))
    #     target_transform.append(
    #         torchvision.transforms.Scale(1024 // downscale,
    #                                      interpolation=Image.NEAREST))
    # transform.extend([
    #     torchvision.transforms.Scale(1024),
    #     net.transform
    #     ])
    # target_transform.extend([
    #     torchvision.transforms.Scale(1024, interpolation=Image.NEAREST),
    #     to_tensor_raw
    #     ])
    # transform = torchvision.transforms.Compose(transform)
    # target_transform = torchvision.transforms.Compose(target_transform)
    

    # datasets = [get_dataset(name, os.path.join(datadir,name), transform=transform,
    #                         target_transform=target_transform)
    #             for name in dataset]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataset_string = ''.join(x for x in str(dataset) if x.isalpha())
    #dataset_string = 'fake_ct'
    
    train_dataset = Importmmwhs(phases=['train'],data=dataset_string,data_dir=datadir)
    val_dataset = Importmmwhs(phases=['val'],data=dataset_string,data_dir=datadir)

    if weights is not None:
        weights = np.loadtxt(weights)
    # opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
    #                       weight_decay=0.0005)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)

    if augmentation:
        collate_fn = lambda batch: augment_collate(batch, crop=crop_size, flip=True)
    else:
        collate_fn = torch.utils.data.dataloader.default_collate
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=4,
                                           collate_fn=collate_fn,
                                           pin_memory=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                           shuffle=False, num_workers=4,
                                           collate_fn=collate_fn,
                                           pin_memory=True)                                       

    iteration = 0
    best_performance = float('inf')
    best_dice = 0
    losses = deque(maxlen=10)
    last_update = 0
    num = 0
    validation_list = np.zeros(50)
    for epoch in range(epochs):
        for im, label in trainloader:
            
            #skip blank labels
            # if(not np.any(label.cpu().detach().numpy())):
            #     # print(label.cpu().detach().numpy())
            #     continue

            # Clear out gradients
            opt.zero_grad()
            
            # if num < 1:
            #     im_show = im[0,0,:,:]
            #     plt.imshow(im_show, cmap='gray')
            #     plt.show()
            #     num = num +1

            # load data/label
            # im = make_variable(im, requires_grad=False)
            im = im.type(torch.FloatTensor).cuda()
            # label = make_variable(label, requires_grad=False)
            label = label.type(torch.LongTensor).cuda()
            
            # forward pass and compute loss
            preds = net(im)
            # print(im.shape)
            loss = supervised_loss(preds, label)

            # backward pass
            loss.backward()
            losses.append(loss.item())

            # step gradients
            opt.step()

            # log results
            if iteration % 10 == 0:
                logging.info('Iteration {}:\t{}'
                                .format(iteration, np.mean(losses)))
                # Use tensorboard to plot
                writer.add_scalar('loss', np.mean(losses), iteration)
            if step is not None and iteration % step == 0:
                logging.info('Decreasing learning rate by 0.1.')
                step_lr(optimizer, 0.1)
            iteration += 1

        loss_val = 0
        Dice1 = 0
        Dice2 = 0
        Dice3 = 0
        Dice4 = 0
        dice1count = 0
        dice2count = 0
        dice3count = 0
        dice4count = 0
        dice1_mean = 0
        dice2_mean = 0
        dice3_mean = 0
        dice4_mean = 0

        for imval, label in valloader:

            net.eval()
            
            # load data/label
            # im = make_variable(im, requires_grad=False)
            imval = imval.type(torch.FloatTensor).cuda()
            # label = make_variable(label, requires_grad=False)
            label = label.type(torch.LongTensor).cuda()
            
            # forward pass and compute loss
            preds = net(imval)
            # print(preds.shape)
            # print(label.shape)
            loss = supervised_loss(preds, label)
            loss_val += loss.item()

            # # Visualising pred vs actual
            _, preds_img = torch.max(preds.data, 1)
            preds_img = preds_img.cpu().data.numpy()
            label_img = label.cpu().data.numpy()
            # print(np.shape(preds_img))
            
            preds_img_slice = preds_img
            label_img_slice = label_img
            for slice in range(0,1):
                preds_img = preds_img_slice[slice,:,:]
                # print(np.shape(label_img))
                label_img = label_img_slice[slice,:,:]
                # plt.imshow(preds_img, cmap='gray')
                # plt.show()
                # plt.imshow(label_img, cmap='gray')
                # plt.show()

                for k in range(0, 5):
                    #print(np.unique(score_s_img))
                    if k in label_img:
                        dice = np.sum(preds_img[label_img == k] == k)*2.0 / (np.sum(preds_img[preds_img == k] == k) + np.sum(label_img[label_img == k] == k))
                        # print("Area:", k, " Dice:", dice, end=" || ")
                        if k == 1:
                            Dice1 = Dice1 + dice
                            dice1count = dice1count + 1
                        if k == 2:
                            Dice2 = Dice2 + dice
                            dice2count = dice2count + 1
                        if k == 3:
                            Dice3 = Dice3 + dice
                            dice3count = dice3count + 1
                        if k == 4:
                            Dice4 = Dice4 + dice
                            dice4count = dice4count + 1

                # print("")
        if (dice1count != 0):
            dice1_mean = Dice1/dice1count
        if (dice2count != 0):
            dice2_mean = Dice2/dice2count
        if (dice3count != 0):
            dice3_mean = Dice3/dice3count
        if (dice4count != 0):
            dice4_mean = Dice4/dice4count
        print("1: ",dice1_mean,"|2: ",dice2_mean,"|3: ",dice3_mean,"|4: ",dice4_mean)
        avg_dice = (dice1_mean+dice2_mean+dice3_mean+dice4_mean)/4

        dice_list = []
        assd_list = []
        net_test = net
        net_test.eval()
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
            with torch.no_grad():
                for i in range(data.shape[0]):
                    batch_img = np.expand_dims(data[..., i].copy(), 0)
                    batch_img = torch.tensor(batch_img[np.newaxis, ...]).type(torch.FloatTensor).to(device)
                    
                    output_res = net_test(batch_img)

                    # if model == 'fcn8s':
                    #     output = output[0]

                    output_res = torch.softmax(output_res, dim=1)
                    tmp_pred = torch.argmax(output_res, dim=1).cpu().numpy()
                       

                    predict[:, :, i] = tmp_pred[0, :, :]

            for c in range(1, num_cls):
                pred_test_data_tr = predict.copy()
                pred_test_data_tr[pred_test_data_tr != c] = 0
                print(np.unique(pred_test_data_tr))

                pred_gt_data_tr = label.copy()
                pred_gt_data_tr[pred_gt_data_tr != c] = 0

                dice_list.append(
                    mmb.dc(pred_test_data_tr, pred_gt_data_tr))
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
        dice_print = np.mean(dice_mean)

        assd_arr = np.reshape(assd_list, [4, -1]).transpose()

        assd_mean = np.mean(assd_arr, axis=1)
        assd_std = np.std(assd_arr, axis=1)

        print('ASSD:')
        print('AA :%.1f(%.1f)' % (assd_mean[3], assd_std[3]))
        print('LAC:%.1f(%.1f)' % (assd_mean[1], assd_std[1]))
        print('LVC:%.1f(%.1f)' % (assd_mean[2], assd_std[2]))
        print('Myo:%.1f(%.1f)' % (assd_mean[0], assd_std[0]))
        print('Mean:%.1f' % np.mean(assd_mean))
        print(best_dice)

        logging.info('Validation loss {} Epoch: {}'.format(loss_val, epoch))
        validation_list[epoch] = loss_val
        #Include DICE performance as selection criteria    
        performance = loss_val/len(valloader)
        if (best_dice < np.mean(dice_mean)):
        #if (best_dice < avg_dice):
            torch.save(net.state_dict(), osp.join(output, 'checkpoint.pth'))
            print('Saved checkpoint:', osp.join(output, 'checkpoint.pth'))
            logging.info("save model to {}".format(osp.join(output, 'checkpoint.pth')))
            last_update = epoch
        # if (last_update < epoch+1):
        #     torch.save(net.state_dict(), osp.join(output, 'checkpoint2.pth'))
        #     print('Saved checkpoint:', osp.join(output, 'checkpoint2.pth'))
        #     logging.info("save model to {}".format(osp.join(output, 'checkpoint2.pth')))
        #     last_update = epoch
        best_performance = min(performance, best_performance)
        best_dice = max(avg_dice,best_dice)
        
        # if epoch - last_update >= 15:
        #     print(f'Optimization complete at epoch {epoch}')
        #     break
        print(epoch)
        print(validation_list)
        #writer.add_hparams({'optimizer':'SGD','Batch Size':batch_size,'model':model,'dataset':str(dataset),'learning_rate':lr,'save_location':osp.join(output, 'checkpoint.pth')},{'avg_dice':best_dice})

if __name__ == '__main__':
    main()
