import os
import numpy as np
import argparse
import time

import torch

from utils.net_utils import adjust_learning_rate, save_checkpoint, clip_gradient
from model.wsddn_vgg16 import WSDDN_VGG16
from datasets.wsddn_dataset import WSDDNDataset
from matplotlib import pyplot as plt
import torch.nn.functional as F
import math


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--net', default='WSDDN_VGG16', type=str)
    parser.add_argument('--start_epoch', help='starting epoch', default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs', help='number of epochs', default=20, type=int)
    parser.add_argument('--disp_interval', help='number of iterations to display loss', default=1000, type=int)
    parser.add_argument('--save_interval', dest='save_interval', help='number of epochs to save', default=1, type=int)
    parser.add_argument('--multiscale', action='store_true')
    parser.add_argument('--save_dir', help='directory to save models', default="../repo/wsddn")
    parser.add_argument('--data_dir', help='directory to load data', default='./data', type=str)

    parser.add_argument('--prop_method', help='ss or eb', default='eb', type=str)
    parser.add_argument('--use_prop_score', action='store_true')
    parser.add_argument('--min_prop', help='minimum proposal box size', default=20, type=int)
    parser.add_argument('--alpha', help='alpha for spatial regularization', default=0.0001, type=float)

    parser.add_argument('--lr', help='starting learning rate', default=0.00001, type=float)
    parser.add_argument('--s', dest='session', help='training session', default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume', help='resume checkpoint or not', action='store_true')
    parser.add_argument('--checksession', dest='checksession', help='checksession to load model', default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch', help='checkepoch to load model', default=1, type=int)
    parser.add_argument('--not_load_optim', dest='no_optim', action='store_true')

    args = parser.parse_args()
    return args


def draw_box(boxes, col=None):
    for j, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        if col is None:
            c = np.random.rand(3)
        else:
            c = col
        plt.hlines(ymin, xmin, xmax, colors=c, lw=2)
        plt.hlines(ymax, xmin, xmax, colors=c, lw=2)
        plt.vlines(xmin, ymin, ymax, colors=c, lw=2)
        plt.vlines(xmax, ymin, ymax, colors=c, lw=2)


def train():
    args = parse_args()
    print('Called with args:')
    print(args)

    np.random.seed(3)
    torch.manual_seed(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(5)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dataset = WSDDNDataset(dataset_names=['voc07_trainval'], data_dir=args.data_dir, prop_method=args.prop_method,
                                 h_flip=True, multi_scale=args.multiscale, min_prop_scale=args.min_prop)

    lr = args.lr

    if args.net == 'WSDDN_VGG16':
        model = WSDDN_VGG16(os.path.join(args.data_dir, 'pretrained_model/vgg16_caffe.pth'), 20)

    else:
        raise Exception('network is not defined')

    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]

    optimizer = torch.optim.SGD(params, momentum=0.9)

    if args.resume:
        load_name = os.path.join(output_dir, '{}_{}_{}.pth'.format(args.net, args.checksession, args.checkepoch))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        assert args.net == checkpoint['net']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("loaded checkpoint %s" % (load_name))

    log_file_name = os.path.join(output_dir, 'log_{}_{}.txt'.format(args.net, args.session))
    if args.resume:
        log_file = open(log_file_name, 'a')
    else:
        log_file = open(log_file_name, 'w')
    log_file.write(str(args))
    log_file.write('\n')

    model.to(device)

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        model.train()
        loss_sum = 0
        reg_sum = 0
        iter_sum = 0
        num_prop = 0
        start = time.time()

        rand_perm = np.random.permutation(len(train_dataset))
        for step in range(1, len(train_dataset) + 1):
            index = rand_perm[step - 1]
            im_data, gt_boxes, box_labels, proposals, prop_scores, image_level_label, im_scale, raw_img, im_id = train_dataset[index]

            # plt.imshow(raw_img)
            # draw_box(proposals / im_scale)
            # draw_box(gt_boxes / im_scale, 'black')
            # plt.show()

            im_data = im_data.unsqueeze(0).to(device)
            rois = proposals.to(device)
            image_level_label = image_level_label.to(device)

            if args.use_prop_score:
                prop_scores = prop_scores.to(device)
            else:
                prop_scores = None
            scores, loss, reg = model(im_data, rois, prop_scores, image_level_label)
            reg = reg * args.alpha
            num_prop += proposals.size(0)
            loss_sum += loss.item()
            reg_sum += reg.item()
            loss = loss + reg

            optimizer.zero_grad()
            loss.backward()
            clip_gradient(model, 10.0)

            optimizer.step()
            iter_sum += 1

            if step % args.disp_interval == 0:
                end = time.time()

                print("[net %s][session %d][epoch %2d][iter %4d] loss: %.4f, reg: %.4f, num_prop: %.1f, lr: %.2e, time: %.1f" %
                      (args.net, args.session, epoch, step, loss_sum / iter_sum,  reg_sum / iter_sum, num_prop / iter_sum, lr,  end - start))
                log_file.write("[net %s][session %d][epoch %2d][iter %4d] loss: %.4f, reg: %.4f, num_prop: %.1f, lr: %.2e, time: %.1f\n" %
                               (args.net, args.session, epoch, step, loss_sum / iter_sum, reg_sum / iter_sum, num_prop / iter_sum, lr,  end - start))
                loss_sum = 0
                reg_sum = 0
                num_prop = 0
                iter_sum = 0
                start = time.time()

        log_file.flush()
        if epoch == 10:
            adjust_learning_rate(optimizer, 0.1)
            lr *= 0.1

        if epoch % args.save_interval == 0:
            save_name = os.path.join(output_dir, '{}_{}_{}.pth'.format(args.net, args.session, epoch))
            checkpoint = dict()
            checkpoint['net'] = args.net
            checkpoint['session'] = args.session
            checkpoint['epoch'] = epoch + 1
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()

            save_checkpoint(checkpoint, save_name)
            print('save model: {}'.format(save_name))

    log_file.close()


if __name__ == '__main__':
    train()