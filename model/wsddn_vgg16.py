import torch.nn as nn
import torch.nn.functional as F
import torch
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.roi_pooling.modules.roi_pool import _RoIPooling
from utils.box_utils import *
import torchvision


class WSDDN_VGG16(nn.Module):
    def __init__(self, pretrained_model_path=None, num_class=20):
        super(WSDDN_VGG16, self).__init__()
        vgg = torchvision.models.vgg16()
        if pretrained_model_path is None:
            print("Create WSDDN_VGG16 without pretrained weights")
        else:
            print("Loading pretrained VGG16 weights from %s" % (pretrained_model_path))
            state_dict = torch.load(pretrained_model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        self.base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        self.top = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        self.num_classes = num_class

        self.fc8c = nn.Linear(4096, self.num_classes)
        self.fc8d = nn.Linear(4096, self.num_classes)
        self.roi_pooling = _RoIPooling(7, 7, 1.0/16.0)
        self.roi_align = RoIAlignAvg(7, 7, 1.0/16.0)
        self.num_classes = self.num_classes
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.fc8c, 0, 0.01, False)
        normal_init(self.fc8d, 0, 0.01, False)

    def forward(self, im_data, rois, prop_scores=None, image_level_label=None):
        N = rois.size(0)
        feature_map = self.base(im_data)
        zero_padded_rois = torch.cat([torch.zeros(N, 1).to(rois), rois], 1)
        pooled_feat = self.roi_pooling(feature_map, zero_padded_rois).view(N, -1)

        if prop_scores is not None:
            pooled_feat = pooled_feat * (prop_scores.view(N, 1) * 10 + 1)

        fc7 = self.top(pooled_feat)
        fc8c = self.fc8c(fc7)
        fc8d = self.fc8d(fc7)

        cls = F.softmax(fc8c, dim=1)
        det = F.softmax(fc8d, dim=0)

        scores = cls * det
        scores = torch.clamp(scores, min=0, max=1)

        if image_level_label is None:
            return scores

        image_level_scores = torch.sum(scores, 0)
        #print(image_level_scores, image_level_label)
        loss = F.binary_cross_entropy(image_level_scores, image_level_label.to(torch.float32))
        reg = self.spatial_regulariser(rois, fc7, scores, image_level_label)

        return scores, loss, reg

    # def spatial_regulariser(self, rois, fc7, scores, image_level_label):
    #     N = rois.size(0)
    #     ret = 0
    #     C = 0
    #     for cls in range(self.num_classes):
    #         if image_level_label[cls].item() == 0:
    #             continue
    #
    #         max_score, max_score_index = torch.max(scores[:, cls], 0)
    #         max_score_box = rois[max_score_index]
    #         max_feature = fc7[max_score_index]
    #
    #         iou = all_pair_iou(max_score_box.view(1, 4), rois).view(N)
    #         adjacent_indices = iou.gt(0.6).nonzero().squeeze()
    #         adjacent_features = fc7[adjacent_indices]
    #
    #         diff = adjacent_features - max_feature
    #         diff = diff * max_score
    #
    #         ret = torch.sum(diff * diff) + ret
    #         C = C + 1
    #     return ret / C

    def spatial_regulariser(self, rois, fc7, scores, image_level_label):
        N = rois.size(0)
        ret = 0
        for cls in range(self.num_classes):
            if image_level_label[cls].item() == 0:
                continue

            max_score, max_score_index = torch.max(scores[:, cls], 0)
            max_score_box = rois[max_score_index]
            max_feature = fc7[max_score_index]

            iou = all_pair_iou(max_score_box.view(1, 4), rois).view(N)
            adjacent_indices = iou.gt(0.6).nonzero().squeeze()
            adjacent_features = fc7[adjacent_indices]

            diff = adjacent_features - max_feature
            diff = diff * max_score

            ret = torch.sum(diff * diff) * 0.5 + ret

        return ret