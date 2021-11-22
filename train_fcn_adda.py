import logging
from matplotlib import pyplot as plt
import os
import os.path as osp
from collections import deque
import itertools
from datetime import datetime

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter

from PIL import Image
from torch.autograd import Variable

import medpy.metric.binary as mmb
import pytz
import time
import sys
sys.path.append('D:\cycada_2\cycada_2')

from cycada.data.mmwhsdata_adda import Importmmwhs_adda
from cycada.models import get_model
from cycada.models.models import models
from cycada.models import VGG16_FCN8s, Discriminator
from cycada.util import config_logging
from cycada.util import to_tensor_raw
from cycada.tools.util import make_variable

BASE_FID = 'D:/mmwhs_sifa/mmwhs_sifa/new_transformed_data/fake_mr/fake_mr_test_npz' # folder path of test files
TESTFILE_FID = 'test_datalist.txt' # path of the .txt file storing the test filenames
TEST_MODALITY = 'CT'
data_size = [256, 256, 1]
label_size = [256, 256, 1]

def check_label(label, num_cls):
    "Check that no labels are out of range"
    label_classes = np.unique(label.numpy().flatten())
    label_classes = label_classes[label_classes < 255]
    if len(label_classes) == 0:
        print('All ignore labels')
        return False
    class_too_large = label_classes.max() > num_cls
    if class_too_large or label_classes.min() < 0:
        print('Labels out of bound')
        print(label_classes)
        return False
    return True



def forward_pass(net, discriminator, im, requires_grad=False, discrim_feat=False):
    if discrim_feat:
        score, feat = net(im)
        dis_score = discriminator(feat)
    else:
        score = net(im)
        dis_score = discriminator(score)
    if not requires_grad:
        score = Variable(score.data, requires_grad=False)
        
    return score, dis_score

def supervised_loss(score, label, weights=None):
    loss_fn_ = torch.nn.NLLLoss(weight=weights, size_average=True, 
            ignore_index=255)
    loss = loss_fn_(F.log_softmax(score, dim=1), label.long())
    return loss
   
def discriminator_loss(score, target_val, lsgan=False):
    if lsgan:
        loss = 0.5 * torch.mean((score - target_val)**2)
    else:
        _,_,h,w = score.size()
        target_val_vec = Variable(target_val * torch.ones(1,h,w),requires_grad=False).long().cuda()
        loss = supervised_loss(score, target_val_vec)
    return loss

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n,n)

def seg_accuracy(score, label, num_cls):
    _, preds = torch.max(score.data, 1)
    hist = fast_hist(label.cpu().numpy().flatten(),
            preds.cpu().numpy().flatten(), num_cls)
    intersections = np.diag(hist)
    unions = (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8) * 100
    acc = np.diag(hist).sum() / hist.sum()
    return intersections, unions, acc

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
@click.argument('output')
@click.option('--dataset', required=True, multiple=True)
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--lr', '-l', default=0.0001)
@click.option('--momentum', '-m', default=0.9)
@click.option('--batch', default=1)
@click.option('--snapshot', '-s', default=5000)
@click.option('--downscale', type=int)
@click.option('--crop_size', default=None, type=int)
@click.option('--half_crop', default=None)
@click.option('--cls_weights', type=click.Path(exists=True))
@click.option('--weights_discrim', type=click.Path(exists=True))
@click.option('--weights_init', type=click.Path(exists=True))
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--lsgan/--no_lsgan', default=False)
@click.option('--num_cls', type=int, default=5)
@click.option('--gpu', default='0')
@click.option('--max_iter', default=10000)
@click.option('--lambda_d', default=1.0)
@click.option('--lambda_g', default=1.0)
@click.option('--train_discrim_only', default=False)
@click.option('--discrim_feat/--discrim_score', default=False)
@click.option('--weights_shared/--weights_unshared', default=False)
def main(output, dataset, datadir, lr, momentum, snapshot, downscale, cls_weights, gpu, 
        weights_init, num_cls, lsgan, max_iter, lambda_d, lambda_g,
        train_discrim_only, weights_discrim, crop_size, weights_shared,
        discrim_feat, half_crop, batch, model):
    


    # So data is sampled in consistent way
    np.random.seed(1337)
    torch.manual_seed(1337)
    logdir = 'runs/{:s}/{:s}_to_{:s}/lr{:.1g}_ld{:.2g}_lg{:.2g}'.format(model, dataset[0],
            dataset[1], lr, lambda_d, lambda_g)
    output = osp.join(output, time.strftime('%y%m%d-%I%M', time.localtime()))
    save_output = output
    if not os.path.exists(output):
        os.mkdir(output)
    if weights_shared:
        logdir += '_weightshared'
    else:
        logdir += '_weightsunshared'
    if discrim_feat:
        logdir += '_discrimfeat'
    else:
        logdir += '_discrimscore'
   # logdir += '/' + datetime.now().strftime('%Y_%b_%d-%H:%M')
    writer = SummaryWriter(log_dir=logdir)

    test_fid = BASE_FID + '/' + TESTFILE_FID
    test_list = read_lists(BASE_FID,test_fid)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_logging()
    print('Train Discrim Only', train_discrim_only)
    net = get_model(model, num_cls=num_cls, pretrained=True, weights_init=weights_init,
            output_last_ft=discrim_feat)
    if weights_shared:
        net_src = net # shared weights
        print("Weights Shared")
    else:
        net_src = get_model(model, num_cls=num_cls, pretrained=True, 
                weights_init=weights_init, output_last_ft=discrim_feat)
        net_src.eval()
        print("Weights Unshared")

    odim = 1 if lsgan else 2
    idim = num_cls if not discrim_feat else 4096
    print('discrim_feat', discrim_feat, idim)
    
    print('discriminator init weights: ', weights_init)
    discriminator = Discriminator(input_dim=idim, output_dim=odim, 
            pretrained=not (weights_discrim==None), 
            weights_init=weights_discrim).cuda()

    # loader = AddaDataLoader(net.transform, dataset, datadir, downscale, 
    #         crop_size=crop_size, half_crop=half_crop,
    #         batch_size=batch, shuffle=True, num_workers=2)
    train_set = Importmmwhs_adda(phases=['train'],data_dir=datadir)
    loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch, num_workers=2, shuffle=True)
    print('dataset', dataset)

    # Class weighted loss?
    if cls_weights is not None:
        weights = np.loadtxt(cls_weights)
    else:
        weights = None
  
    # setup optimizers
    opt_dis = torch.optim.SGD(discriminator.parameters(), lr=lr, 
            momentum=momentum, weight_decay=0.0005)
    opt_rep = torch.optim.SGD(net.parameters(), lr=lr, 
            momentum=momentum, weight_decay=0.0005)

    iteration = 0
    num_update_g = 0
    last_update_g = -1
    losses_super_s = deque(maxlen=100)
    losses_super_t = deque(maxlen=100)
    losses_dis = deque(maxlen=100)
    losses_rep = deque(maxlen=100)
    accuracies_dom = deque(maxlen=100)
    intersections = np.zeros([100,num_cls])
    unions = np.zeros([100, num_cls])
    accuracy = deque(maxlen=100)
    print('max iter:', max_iter)
   
    net.train()
    discriminator.train()

    best_dice = 0
    best_iteration = 0
    while iteration < max_iter:
        
        for im_s, im_t, label_s, label_t in loader:
            
            if iteration > max_iter:
                break
           
            info_str = 'Iteration {}: '.format(iteration)
            
            if not check_label(label_s, num_cls):
                continue
            
            ###########################
            # 1. Setup Data Variables #
            ###########################
            im_s = make_variable(im_s, requires_grad=False)
            label_s = make_variable(label_s, requires_grad=False)
            # label_s = label_s.type(torch.LongTensor).cuda()
            im_t = make_variable(im_t, requires_grad=False)
            # label_t = make_variable(label_t, requires_grad=False)
            label_t = label_t.type(torch.LongTensor).cuda()
           
            #############################
            # 2. Optimize Discriminator #
            #############################
            
            # zero gradients for optimizer
            opt_dis.zero_grad()
            opt_rep.zero_grad()
            
            # extract features
            if discrim_feat:
                score_s, feat_s = net_src(im_s)
                score_s = Variable(score_s.data, requires_grad=False)
                f_s = Variable(feat_s.data, requires_grad=False)
            else:
                score_s = Variable(net_src(im_s).data, requires_grad=False)
                f_s = score_s
            dis_score_s = discriminator(f_s)
            

            
            
            if discrim_feat:
                score_t, feat_t = net(im_t)
                score_t = Variable(score_t.data, requires_grad=False)
                f_t = Variable(feat_t.data, requires_grad=False)
            else:
                score_t = Variable(net(im_t).data, requires_grad=False)
                f_t = score_t
            dis_score_t = discriminator(f_t)
            
            dis_pred_concat = torch.cat((dis_score_s, dis_score_t))


            # prepare real and fake labels
            batch_t,_,h,w = dis_score_t.size()
            batch_s,_,_,_ = dis_score_s.size()
            dis_label_concat = make_variable(
                    torch.cat(
                        [torch.ones(batch_s,h,w).long(), 
                        torch.zeros(batch_t,h,w).long()]
                        ), requires_grad=False)

            # compute loss for discriminator
            loss_dis = supervised_loss(dis_pred_concat, dis_label_concat)
            (lambda_d * loss_dis).backward()
            losses_dis.append(loss_dis.item())

            # optimize discriminator
            opt_dis.step()

            # compute discriminator acc
            pred_dis = torch.squeeze(dis_pred_concat.max(1)[1])
            dom_acc = (pred_dis == dis_label_concat).float().mean().item() 
            accuracies_dom.append(dom_acc * 100.)

            # add discriminator info to log
            info_str += " domacc:{:0.1f}  D:{:.3f}".format(np.mean(accuracies_dom), 
                    np.mean(losses_dis))
            writer.add_scalar('loss/discriminator', np.mean(losses_dis), iteration)
            writer.add_scalar('acc/discriminator', np.mean(accuracies_dom), iteration)

            ###########################
            # Optimize Target Network #
            ###########################
            
            # print('acc/discrm', np.mean(accuracies_dom))
            
           
            dom_acc_thresh = 51
            # 
            if not train_discrim_only and np.mean(accuracies_dom) > dom_acc_thresh:
              
                last_update_g = iteration
                num_update_g += 1 
                if num_update_g % 1 == 0:
                    print('Updating G with adversarial loss ({:d} times)'.format(num_update_g))

                # zero out optimizer gradients
                opt_dis.zero_grad()
                opt_rep.zero_grad()

                # extract features
                if discrim_feat:
                    score_t, feat_t = net(im_t)
                    score_t = Variable(score_t.data, requires_grad=False)
                    f_t = feat_t 
                else:
                    score_t = net(im_t)
                    f_t = score_t

                #score_t = net(im_t)
                dis_score_t = discriminator(f_t)
                #print(dis_score_t)

                # create fake label
                batch,_,h,w = dis_score_t.size()
                target_dom_fake_t = make_variable(torch.ones(batch,h,w).long(), 
                        requires_grad=False)

                # compute loss for target net
                loss_gan_t = supervised_loss(dis_score_t, target_dom_fake_t)
                (lambda_g * loss_gan_t).backward()
                losses_rep.append(loss_gan_t.item())
                writer.add_scalar('loss/generator', np.mean(losses_rep), iteration)
                
                # optimize target net
                opt_rep.step()

                # log net update info
                info_str += ' G:{:.3f}'.format(np.mean(losses_rep))
               
            if (not train_discrim_only) and weights_shared and (np.mean(accuracies_dom) > dom_acc_thresh) and False:
               
                print('Updating G using source supervised loss.')

                # zero out optimizer gradients
                opt_dis.zero_grad()
                opt_rep.zero_grad()

                # extract features
                if discrim_feat:
                    score_s, _ = net(im_s)
                else:
                    score_s = net(im_s)

                loss_supervised_s = supervised_loss(score_s, label_s, 
                        weights=weights)
                loss_supervised_s.backward()
                losses_super_s.append(loss_supervised_s.item())
                info_str += ' clsS:{:.2f}'.format(np.mean(losses_super_s))
                writer.add_scalar('loss/supervised/source', np.mean(losses_super_s), iteration)

                # optimize target net
                opt_rep.step()

            # compute supervised losses for target -- monitoring only!!!
            loss_supervised_t = supervised_loss(score_t, label_t, weights=weights)
            losses_super_t.append(loss_supervised_t.item())
            info_str += ' clsT:{:.2f}'.format(np.mean(losses_super_t))
            writer.add_scalar('loss/supervised/target', np.mean(losses_super_t), iteration)

            ###########################
            # Log and compute metrics #
            ###########################
            if iteration % 100 == 0 and iteration > 0:
                
                # compute metrics
                intersection,union,acc = seg_accuracy(score_t, label_t.data, num_cls) 
                intersections = np.vstack([intersections[1:,:], intersection[np.newaxis,:]])
                unions = np.vstack([unions[1:,:], union[np.newaxis,:]]) 
                accuracy.append(acc.item() * 100)
                acc = np.mean(accuracy)
                mIoU =  np.mean(np.maximum(intersections, 1) / np.maximum(unions, 1)) * 100

                #Include DICE as a metric/indicator
                im_t_show = im_t.cpu().data.numpy()
                im_t_show = im_t_show[0,0,:,:]
                # plt.imshow(im_t_show, cmap='gray')
                # plt.show()

                _, preds = torch.max(score_t.data, 1)
                score_t_img = preds.cpu().data.numpy()
                # print(np.shape(score_t_img))
                score_t_img = score_t_img[0,:,:]
                # plt.imshow(score_t_img, cmap='gray')
                # plt.show()
                
                # print(type(label_t_img))
                label_t_img = label_t.cpu().data.numpy()
                # print(np.shape(label_t_img))
                label_t_img = label_t_img[0,:,:]
                # plt.imshow(label_t_img, cmap='gray')
                # plt.show()

                dice_list = []
                assd_list = []
                net_test = net
                net_test.eval()
                for idx_file, fid in enumerate(test_list):
                    _npz_dict = np.load(fid)
                    data = _npz_dict['arr_0']
                    label = _npz_dict['arr_1']
                    predict = np.zeros(label.shape)

                    # data = (data - data.mean())/data.std()

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
                            batch_img = torch.tensor(batch_img[np.newaxis, ...]).type(
                                torch.FloatTensor).to(device)
                            #print(batch_img.shape)

                            output = net_test(batch_img)
                            
                            if (model == 'fcn8s' and discrim_feat == True):
                                output = output[0]

                            output = torch.softmax(output, dim=1)
                            tmp_pred = torch.argmax(output, dim=1).cpu().numpy()
                            # print(output.shape,tmp_pred.shape)

                            predict[:, :, i] = tmp_pred[0, :, :]

                            # if i > 152:
                            #     tmp_pred2 = tmp_pred[0,:,:]
                            #     plt.imshow(tmp_pred2, cmap='gray')
                            #     plt.show()
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

                info_str += ' acc:{:0.2f}  mIoU:{:0.2f}'.format(acc, mIoU)
                writer.add_scalar('metrics/acc', np.mean(accuracy), iteration)
                writer.add_scalar('metrics/mIoU', np.mean(mIoU), iteration)
                logging.info(info_str)

                #save best dice
                if best_dice < np.mean(dice_mean):
                    output = save_output
                    best_dice = np.mean(dice_mean)
                    best_iteration = iteration
                    os.makedirs(output, exist_ok=True)
                    if not train_discrim_only:
                        torch.save(net.state_dict(),
                                '{}/bestnet-iter{}.pth'.format(output,iteration))
                    torch.save(discriminator.state_dict(),
                            '{}/bestdiscriminator-iter.pth'.format(output))

                print(best_dice)
                print(best_iteration)   
            iteration += 1

            ################
            # Save outputs #
            ################
            output = save_output
            # every 100 iters save current model
            if iteration % 100 == 0:
                os.makedirs(output, exist_ok=True)
                if not train_discrim_only:
                    torch.save(net.state_dict(),
                            '{}/net-itercurr.pth'.format(output))
                torch.save(discriminator.state_dict(),
                        '{}/discriminator-itercurr.pth'.format(output))

            # save labeled snapshots
            if iteration % snapshot == 0:
                os.makedirs(output, exist_ok=True)
                if not train_discrim_only:
                    torch.save(net.state_dict(),
                            '{}/net-iter{}.pth'.format(output, iteration))
                torch.save(discriminator.state_dict(),
                        '{}/discriminator-iter{}.pth'.format(output, iteration))

            if iteration - last_update_g >= len(loader):
                print('No suitable discriminator found -- returning.')
                os.makedirs(output, exist_ok=True)
                torch.save(net.state_dict(),
                            osp.join(output,'iter{}.pth'.format(iteration)))
                iteration = max_iter # make sure outside loop breaks
                break

   # writer.add_hparams({'optimizer':'SGD','Batch Size':batch_size,'model':model,'dataset':str(dataset),'learning_rate':lr},{'avg_dice':best_dice})
    writer.close()


if __name__ == '__main__':
    main()
