# --------------------------------------------------------
# PyTorch WSDDN
# Licensed under The MIT License [see LICENSE for details]
# Written by Seungkwan Lee
# Some parts of this implementation are based on code from Ross Girshick, Jiasen Lu, and Jianwei Yang
# --------------------------------------------------------
import os
import numpy as np
import argparse
import time

import torch

from model.wsddn_vgg16 import WSDDN_VGG16
from datasets.wsddn_dataset import WSDDNDataset
from matplotlib import pyplot as plt
import torch.nn.functional as F
import math
import pickle
from utils.cpu_nms import cpu_nms as nms
import heapq

from frcnn_eval.pascal_voc import voc_eval_kit

def parse_args():
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('--save_dir', help='directory to load model and save detection results', default="../repo")
    parser.add_argument('--data_dir', help='directory to load data', default='./data', type=str)

    parser.add_argument('--prop_method', help='ss or eb', default='eb', type=str)
    parser.add_argument('--use_prop_score', action='store_true')
    parser.add_argument('--min_prop', help='minimum proposal box size', default=20, type=int)
    parser.add_argument('--model_name', default='WSDDN_VGG16_1_20', type=str)

    args = parser.parse_args()
    return args

args = parse_args()

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


def eval():
    print('Called with args:')
    print(args)

    np.random.seed(3)
    torch.manual_seed(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(5)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    eval_kit = voc_eval_kit('test', '2007', os.path.join(args.data_dir, 'VOCdevkit2007'))

    test_dataset = WSDDNDataset(dataset_names=['voc07_test'], data_dir=args.data_dir, prop_method=args.prop_method,
                                num_classes=20, min_prop_scale=args.min_prop)

    load_name = os.path.join(args.save_dir, 'wsddn', '{}.pth'.format(args.model_name))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    if checkpoint['net'] == 'WSDDN_VGG16':
        model = WSDDN_VGG16(None, 20)
    else:
        raise Exception('network is not defined')
    model.load_state_dict(checkpoint['model'])
    print("loaded checkpoint %s" % (load_name))

    model.to(device)
    model.eval()

    start = time.time()

    num_images = len(test_dataset)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = 40 * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(20)
    # thresh = 0.1 * np.ones(imdb.num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in range(20)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(20)]

    for index in range(len(test_dataset)):
        im_data, gt_boxes, box_labels, proposals, prop_scores, image_level_label, im_scale, raw_img, im_id = test_dataset.get_data(index, False, 688)

        #print(image_level_label)
        #plt.imshow(raw_img)
        # draw_box(proposals / im_scale)
        # draw_box(gt_boxes / im_scale, 'black')
        #plt.show()

        im_data = im_data.unsqueeze(0).to(device)
        rois = proposals.to(device)

        if args.use_prop_score:
            prop_scores = prop_scores.to(device)
        else:
            prop_scores = None
        scores = model(im_data, rois, prop_scores, None).detach().cpu().numpy()
        # print(scores.max())
        # print(np.sum(scores, axis=0))
        # print(np.argmax(np.sum(scores, axis=0)))
        boxes = proposals.numpy() / im_scale

        for cls in range(20):
            inds = np.where((scores[:, cls] > thresh[cls]))[0]
            cls_scores = scores[inds, cls]
            cls_boxes = boxes[inds].copy()
            top_inds = np.argsort(-cls_scores)[:max_per_image]
            cls_scores = cls_scores[top_inds]
            cls_boxes = cls_boxes[top_inds, :]

            # if cls_scores[0] > 0.001:
            #     #print(cls)
            #     plt.imshow(raw_img)
            #     draw_box(cls_boxes[0:10, :])
            #     draw_box(gt_boxes / im_scale, 'black')
            #     plt.show()

            # push new scores onto the minheap
            for val in cls_scores:
                heapq.heappush(top_scores[cls], val)
            # if we've collected more than the max number of detection,
            # then pop items off the minheap and update the class threshold
            if len(top_scores[cls]) > max_per_set:
                while len(top_scores[cls]) > max_per_set:
                    heapq.heappop(top_scores[cls])
                thresh[cls] = top_scores[cls][0]

            all_boxes[cls][index] = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)

        # sorted_scores, sorted_indices = torch.sort(scores.detach(), dim=0, descending=True)
        # sorted_boxes = rois[sorted_indices.permute(1, 0)]
        #
        # for cls in range(20):
        #     here = torch.cat((sorted_boxes[cls], sorted_scores[:, cls:cls + 1]), 1).cpu()
        #     print(here)
        #     all_boxes[cls][index] = here.numpy()

        if index % 100 == 99:
           print('%d images complete, elapsed time:%.1f' % (index + 1, time.time() - start))

    for j in range(20):
        for i in range(len(test_dataset)):
            inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]

    save_name = os.path.join(args.save_dir, 'detection_result', '{}.pkl'.format(args.model_name))
    pickle.dump(all_boxes, open(save_name, 'wb'))

    print('Detection Complete, elapsed time: %.1f', time.time() - start)

    for cls in range(20):
        for index in range(len(test_dataset)):
            dets = all_boxes[cls][index]
            if dets == []:
                continue
            keep = py_cpu_nms(dets, 0.4)
            all_boxes[cls][index] = dets[keep, :].copy()
    print('NMS complete, elapsed time: %.1f', time.time() - start)

    eval_kit.evaluate_detections(all_boxes)


def my_eval():
    print('Called with args:')
    print(args)

    np.random.seed(3)
    torch.manual_seed(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(5)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    eval_kit = voc_eval_kit('test', '2007', os.path.join(args.data_dir, 'VOCdevkit2007'))

    test_dataset = WSDDNDataset(dataset_names=['voc07_test'], data_dir=args.data_dir, prop_method=args.prop_method,
                                num_classes=20, min_prop_scale=args.min_prop)

    load_name = os.path.join(args.save_dir, 'wsddn', '{}.pth'.format(args.model_name))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    if checkpoint['net'] == 'WSDDN_VGG16':
        model = WSDDN_VGG16(None, 20)
    else:
        raise Exception('network is not defined')
    model.load_state_dict(checkpoint['model'])
    print("loaded checkpoint %s" % (load_name))

    model.to(device)
    model.eval()

    start = time.time()

    all_boxes = [[[] for _ in range(len(test_dataset))] for _ in range(20)]

    for index in range(len(test_dataset)):
        im_data, gt_boxes, box_labels, proposals, prop_scores, image_level_label, im_scale, raw_img, im_id = test_dataset.get_data(
            index, False, 688)

        im_data = im_data.unsqueeze(0).to(device)
        rois = proposals.to(device)

        if args.use_prop_score:
            prop_scores = prop_scores.to(device)
        else:
            prop_scores = None
        scores = model(im_data, rois, prop_scores, None)

        sorted_scores, sorted_indices = torch.sort(scores.detach() * 100, dim=0, descending=True)
        sorted_boxes = rois[sorted_indices.permute(1, 0)] / im_scale

        for cls in range(20):
            here = torch.cat((sorted_boxes[cls], sorted_scores[:, cls:cls + 1]), 1).cpu()
            all_boxes[cls][index] = here.numpy()

        if index % 500 == 499:
            print('%d images complete, elapsed time:%.1f' % (index + 1, time.time() - start))

    save_name = os.path.join(args.save_dir, 'detection_result', '{}.pkl'.format(args.model_name))
    pickle.dump(all_boxes, open(save_name, 'wb'))

    print('Detection Complete, elapsed time: %.1f', time.time() - start)

    for cls in range(20):
        for index in range(len(test_dataset)):
            dets = all_boxes[cls][index]
            if dets == []:
                continue
            keep = nms(dets, 0.4)
            all_boxes[cls][index] = dets[keep, :].copy()
    print('NMS complete, elapsed time: %.1f', time.time() - start)

    eval_kit.evaluate_detections(all_boxes)

def eval_saved_result():
    eval_kit = voc_eval_kit('test', '2007', os.path.join(args.data_dir, 'VOCdevkit2007'))

    save_name = os.path.join(args.save_dir, 'detection_result', '{}.pkl'.format(args.model_name))

    all_boxes = pickle.load(open(save_name, 'rb'), encoding='latin1')
    #all_boxes = pickle.load(open('../repo/oicr_result/test_detections.pkl', 'rb'), encoding='latin1')

    for cls in range(20):
        for index in range(len(all_boxes[0])):
            dets = all_boxes[cls][index]
            if dets == []:
                continue
            keep = nms(dets, 0.4)
            all_boxes[cls][index] = dets[keep, :].copy()
            if index % 500 == 499:
                print(index)
        print('nms: cls %d complete' % cls)

    eval_kit.evaluate_detections(all_boxes)


if __name__ == '__main__':
    my_eval()
    #eval_saved_result()