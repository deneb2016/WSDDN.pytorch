from scipy.misc import imread
from scipy.io import loadmat
import numpy as np
import sys
import os
import xml.etree.ElementTree as ET

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']


class VOCLoader:
    def __init__(self, root, prop_method, year, name):
        self.items = []
        self.name_to_index = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        print('VOC %s %s dataset loading...' % (year, name))

        proposals = {}
        if prop_method == 'eb':
            raw_data = loadmat(os.path.join(root, 'proposals', 'edge_boxes_voc_%s_%s.mat' % (year, name)))
            for i in range(len(raw_data['images'][0])):
                id = raw_data['images'][0][i][0]
                boxes = raw_data['boxes'][0][i].astype(np.float) - 1
                proposals[id] = np.concatenate([boxes[:, 1:2], boxes[:, 0:1], boxes[:, 3:4], boxes[:, 2:3]], 1)
        elif prop_method == 'ss':
            raw_data = loadmat(os.path.join(root, 'proposals', 'selective_search_voc_%s_%s.mat' % (year, name)))
            for i in range(len(raw_data['images'])):
                id = raw_data['images'][i][0][0]
                boxes = raw_data['boxes'][0][i].astype(np.float) - 1
                proposals[id] = np.concatenate([boxes[:, 1:2], boxes[:, 0:1], boxes[:, 3:4], boxes[:, 2:3]], 1)

        rootpath = os.path.join(root, 'VOCdevkit2007', 'VOC' + year)
        for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
            data = {}
            id = line.strip()
            target = ET.parse(os.path.join(rootpath, 'Annotations', id + '.xml'))

            box_set = []
            category_set = []
            for obj in target.iter('object'):
                cls_name = obj.find('name').text.strip().lower()
                bbox = obj.find('bndbox')

                xmin = int(bbox.find('xmin').text) - 1
                ymin = int(bbox.find('ymin').text) - 1
                xmax = int(bbox.find('xmax').text) - 1
                ymax = int(bbox.find('ymax').text) - 1

                category = self.name_to_index[cls_name]
                box_set.append(np.array([xmin, ymin, xmax, ymax], np.float32))
                category_set.append(category)

            data['id'] = id
            data['boxes'] = np.array(box_set)
            data['categories'] = np.array(category_set, np.long)
            data['img_path'] = os.path.join(rootpath, 'JPEGImages', line.strip() + '.jpg')
            data['proposals'] = proposals[id]
            self.items.append(data)

        print('VOC %s %s dataset loading complete' % (year, name))

    def __len__(self):
        return len(self.items)
