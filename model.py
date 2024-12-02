import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import BasicBlock, Bottleneck, ResNet


class Facial(nn.Module):
    def __init__(self, num_classes=1000, teacher='resnet18', students='resnet18'):
        super(Facial, self).__init__()
        self.num_class = num_classes

        # 定义teacher和students模型结构，默认为ResNet18
        if teacher == 'resnet50':
            self.out_channel = 2048
            self.teacher_backbone = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
        else:
            self.out_channel = 512
            self.teacher_backbone = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

        if students == 'resnet50':
            self.in_channel = 2048
            self.students_backbone = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
        else:
            self.in_channel = 512
            self.students_backbone = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

        # 通道对齐
        self.channel_mapping = nn.Sequential(
            # nn.ReplicationPad2d(1),
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU())

        # 构建特征融合
        self.feature_fusion = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(self.in_channel + self.out_channel, self.out_channel, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU())

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 构建分类器
        self.classifier_t = nn.Linear(self.out_channel, num_classes, bias=False)
        self.classifier_s0 = nn.Linear(self.out_channel, num_classes, bias=False)
        self.classifier_s1 = nn.Linear(self.out_channel, num_classes, bias=False)

        # 构建教师模型分类器
        self.classifier_ts = nn.Linear(self.out_channel, num_classes, bias=False)
        self.classifier_ts.weight.requires_grad = False
        self.classifier_ts.weight.data = self.classifier_t.weight.data

    def forward(self, inputs, mode):
        assert mode == 'train' or 'val', 'mode is false!'
        if mode == 'val':

            # 教师模型评估输出
            # t_map, t_vector = self.teacher_backbone(inputs)
            # logit_t = self.classifier_t(t_vector)

            # 学生模型评估输出
            s0_map, s0_vector = self.students_backbone(inputs)
            s0_map_mapping = self.channel_mapping(s0_map)
            s0_map_cat = torch.cat((s0_map, s0_map_mapping), dim=1)
            s0_map_fuse = self.feature_fusion(s0_map_cat)
            s0_vector_last = self.avgpool(s0_map_fuse).flatten(1)
            logit_s0 = self.classifier_s0(s0_vector_last)

            return logit_s0, s0_vector_last
        else:

            # inputs[:, 0]是原图， inputs[:, 1]是掩膜图像
            # print(inputs.shape)
            t_map, t_vector = self.teacher_backbone(inputs[:, 0])
            s0_map, s0_vector = self.students_backbone(inputs[:, 0])
            s1_map, s1_vector = self.students_backbone(inputs[:, 1])

            # 教师模型分支
            logit_t = self.classifier_t(t_vector)

            # 学生模型，掩膜分支
            s1_map_mapping = self.channel_mapping(s1_map)
            s1_vector_last = self.avgpool(s1_map_mapping).flatten(1)
            logit_s1 = self.classifier_s1(s1_vector_last)

            # 学生模型，原始图像分支
            s0_map_mapping = self.channel_mapping(s0_map)
            s0_map_cat = torch.cat((s0_map, s0_map_mapping), dim=1)
            s0_map_fuse = self.feature_fusion(s0_map_cat)
            s0_vector_last = self.avgpool(s0_map_fuse).flatten(1)
            logit_s0 = self.classifier_s0(s0_vector_last)

            # CAM，保留掩膜图像分支分类器参数
            # name = list(self.named_parameters())
            # print(name[-3])
            params = list(self.parameters())
            classifier_s1_weights = params[-4].data
            classifier_s1_weights = classifier_s1_weights.view(1, self.num_class, self.out_channel, 1,
                                                               1).clone().detach()

            t_map_copy = t_map.clone().detach()
            t_vector_copy = t_vector.clone().detach()
            s1_map_cam = (s1_map_mapping.unsqueeze(1) * classifier_s1_weights).sum(2).mean(1)
            t_map_cam = (t_map_copy.unsqueeze(1) * classifier_s1_weights).sum(2).mean(1)

            s1_relation = torch.mm(s1_vector_last, s1_vector_last.t())
            t_relation = torch.mm(t_vector_copy, t_vector_copy.t())
            s1_relation_norm = F.normalize(s1_relation, p=2, dim=1)
            t_relation_norm = F.normalize(t_relation, p=2, dim=1)

            logit_t_t = logit_t.clone().detach()
            self.classifier_ts.weight.data = self.classifier_t.weight.data.clone().detach()
            logit_ts_s1 = self.classifier_ts(s1_vector_last)
            logit_m = logit_t_t
            logit_m_ = logit_ts_s1

            logit_t_s0 = self.classifier_t(s0_vector_last)
            logit_s0_t = self.classifier_s0(t_vector)
            logit_n = torch.cat((logit_t, logit_s0), 0)
            logit_n_ = torch.cat((logit_t_s0, logit_s0_t), 0)

            return logit_t, logit_s0, logit_s1, t_map_cam, s1_map_cam, t_relation_norm, s1_relation_norm, logit_m, logit_m_, logit_n, logit_n_

