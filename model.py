
import torch
import torch.nn as nn
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
        self.relation_mapping = nn.Sequential(nn.ReplicationPad2d(1),
                                              nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=1, bias=True),
                                              nn.BatchNorm2d(self.out_channel),
                                              nn.ReLU())

        # 构建特征融合模块
        self.feature_fusion = nn.Sequential(nn.ReplicationPad2d(1),
                                            nn.Conv2d(self.in_channel + self.out_channel, self.out_channel, kernel_size=3, stride=1, bias=True),
                                            nn.BatchNorm2d(self.out_channel),
                                            nn.ReLU(),
                                            nn.AdaptiveAvgPool2d((1, 1)))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 构建分类器
        self.fc_t = nn.Linear(self.out_channel, num_classes)
        self.fc_b1 = nn.Linear(self.out_channel, num_classes)
        self.fc_b2 = nn.Linear(self.out_channel, num_classes)

    def relation_learn(self, x_map):

        x_map_learn = self.relation_mapping(x_map)

        return x_map_learn

    def forward(self, inputs, mode):
        assert mode == 'train' or 'val', 'mode is false!'
        if mode == 'val':

            # 教师模型评估输出
            xt_map, xt_vector = self.teacher_backbone(inputs)
            outs_t = self.fc_t(xt_vector)

            # 学生模型评估输出
            xs_b1_map, xs_b1_vector = self.students_backbone(inputs)
            xs_b1_map_learn = self.relation_learn(xs_b1_map)
            xs_b1_map_last = torch.cat((xs_b1_map, xs_b1_map_learn), dim=1)
            xs_b1_map_last = self.feature_fusion(xs_b1_map_last)
            xs_b1_vector_last = xs_b1_map_last.flatten(1)
            outs_s_b1 = self.fc_b1(xs_b1_vector_last)

            return outs_t, outs_s_b1
        else:

            # inputs[:, 0]是原图， inputs[:, 1]是掩膜图像
            xt_map, xt_vector = self.teacher_backbone(inputs[:, 0])
            xs_b1_map, xs_b1_vector = self.students_backbone(inputs[:, 0])
            xs_b2_map, xs_b2_vector = self.students_backbone(inputs[:, 1])

            # 教师模型分支
            outs_t = self.fc_t(xt_vector)

            # 学生模型，掩膜分支
            xs_b2_map_learn = self.relation_learn(xs_b2_map)
            xs_b2_vector_last = self.avgpool(xs_b2_map_learn).flatten(1)
            outs_s_b2 = self.fc_b2(xs_b2_vector_last)

            # 学生模型，原始图像分支
            xs_b1_map_learn = self.relation_learn(xs_b1_map)
            xs_b1_map_last = torch.cat((xs_b1_map, xs_b1_map_learn), dim=1)
            xs_b1_map_last = self.feature_fusion(xs_b1_map_last)
            xs_b1_vector_last = xs_b1_map_last.flatten(1)
            outs_s_b1 = self.fc_b1(xs_b1_vector_last)

            # CAM，保留掩膜图像分支分类器参数
            params = list(self.parameters())
            fc_b2_weights = params[-2].data
            fc_b2_weights = fc_b2_weights.view(1, self.num_class, self.out_channel, 1, 1).clone().detach()

            return outs_t, xt_vector, xt_map, outs_s_b1, outs_s_b2, xs_b2_map_learn, xs_b2_vector, fc_b2_weights

