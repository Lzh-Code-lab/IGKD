import os
import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from model import Facial
from lossfunc import loss_ce, loss_kd, loss_relation
from utils import *
from data_load import DataLoad, DataErase

#  -- 表示可选参数，无 -- 表示必需参数
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='./data_set/RAF-DB', help='path to dataset')
parser.add_argument('--num-classes', default=7, type=int, help='num of classes')
parser.add_argument('--s_checkpoint', default='./checkpoints/resnet18_msceleb.pth')
parser.add_argument('--t_checkpoint', default='./checkpoints/resnet18_msceleb.pth')
parser.add_argument('--evaluate', default=False, help='evaluate model on validation set')
parser.add_argument('--epochs', default=60, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch-size', default=256, type=int, help='train model batch size')
parser.add_argument('--val-batch-size', default=256, type=int, help='val model batch size')
parser.add_argument('--teacher-backbone', default='resnet18', help='teacher backbone')
parser.add_argument('--students-backbone', default='resnet18', help='students backbone')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--gpu', default=True, help='GPU id to use.')
parser.add_argument('--save-path', default='./logfile', type=str,
                    help='the log and model files save path. Just a string.')
parser.add_argument('--tag', default='RAF-DB', type=str,
                    help='the tag for identifying the log and model files. Just a string.')


def main():
    args = parser.parse_args()
    # gpu or cpu
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'
    print("using {} device.".format(device))
    print('Using {} dataloader workers every process'.format(args.workers))

    # 数据集加载和预处理
    data_loader = DataLoad(args.data, args.train_batch_size, args.val_batch_size, args.workers).to(device)
    data_erasing = DataErase().to(device)

    # 建立教师模型和学生模型
    model = Facial(args.num_classes, args.teacher_backbone, args.students_backbone).to(device)
    count_confident = nn.Sequential(nn.Linear(2*args.num_classes, args.num_classes),
                                    nn.ReLU(),
                                    nn.Linear(args.num_classes, 1),
                                    nn.Sigmoid())
    count_confident.to(device)

    # 构建优化器
    optimizer = optim.Adam([{'params': model.teacher_backbone.parameters(), 'lr': 0.001},
                            {'params': model.students_backbone.parameters(), 'lr': 0.001},
                            {'params': model.relation_mapping.parameters(), 'lr': 0.0002},
                            {'params': model.feature_fusion.parameters(), 'lr': 0.0002},
                            {'params': model.fc_t.parameters(), 'lr': 0.001},
                            {'params': model.fc_b1.parameters(), 'lr': 0.001},
                            {'params': model.fc_b2.parameters(), 'lr': 0.001},
                            {'params': count_confident.parameters(), 'lr': 0.001}], weight_decay=1e-4)

    # 学习率下降
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    # 加载预训练模型
    checkpoint_load(model, args.t_checkpoint, args.s_checkpoint, device)

    # 随机种子
    setup_seed(0)

    # 模型训练和评估
    if args.evaluate:
        val_loader, val_num = data_loader('val')
        print('val dataset num is {}'.format(val_num))
        val_accuracy_t, val_accuracy_s = validate(val_loader, model, device)
        print('teacher model val acc is {:.5f}, students model val acc is {:.5f}'.format(val_accuracy_t, val_accuracy_s))
    else:
        best_acc = 0.0
        train_loader, val_loader, train_num, val_num = data_loader('train')
        for epoch in range(args.start_epoch, args.epochs):
            best_acc = train(train_loader, val_loader, model, count_confident, optimizer, data_erasing, epoch, best_acc, device, args)
            scheduler.step()
        print('Finished Training！Student model best accurate is {:.5f}'.format(best_acc))


def validate(val_loader, model, device):

    model.eval()
    accuracy_t, accuracy_s, val_num = 0.0, 0.0, 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            val_num += val_labels.shape[0]

            outs_t, outs_s_b1 = model(val_images.to(device), mode='val')
            predict_t = outs_t.softmax(1).max(1)[1]
            predict_s = outs_s_b1.softmax(1).max(1)[1]
            accuracy_t += torch.eq(predict_t, val_labels.to(device)).sum().item()
            accuracy_s += torch.eq(predict_s, val_labels.to(device)).sum().item()
            # val_bar.desc = "validate accurate is [{}]".format(acc_v / num_v)
    val_accuracy_t = accuracy_t / val_num
    val_accuracy_s = accuracy_s / val_num
    return val_accuracy_t, val_accuracy_s


def train(train_loader, val_loader, model, count_confident, optimizer, data_erasing, epoch, best_acc, device, args):

    model.train()
    accuracy_t, accuracy_s, running_loss, train_batch_num = 0.0, 0.0, 0.0, 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):

        # 数据选择
        images_origin, labels = data
        train_batch_num += labels.shape[0]

        # 数据处理
        images_erasing = data_erasing(images_origin).to(device)

        # 模型训练输出
        outs_t, xt_vector, xt_map, outs_s_b1, outs_s_b2, xs_b2_map_last, xs_b2_vector_last, fc_b2_weights = \
            model(images_erasing, mode='train')

        # 损失计算和反向回传
        loss_1 = loss_ce(outs_t, outs_s_b1, outs_s_b2, labels, device)
        loss_2 = loss_relation(outs_t, outs_s_b1, xt_vector, xt_map, xs_b2_map_last, xs_b2_vector_last, fc_b2_weights, device)
        loss_3 = loss_kd(outs_t, outs_s_b2, device)
        loss = loss_1 + loss_2 + loss_3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计数据
        running_loss += loss.item()
        predict_t = outs_t.softmax(1).max(1)[1]
        predict_s = outs_s_b1.softmax(1).max(1)[1]
        accuracy_t += torch.eq(predict_t, labels.to(device)).sum().item()
        accuracy_s += torch.eq(predict_s, labels.to(device)).sum().item()
        train_bar.desc = "train epoch[{}/{}] lr:{} loss:{:.3f}".format(epoch + 1, args.epochs,
                                                                       optimizer.param_groups[0]['lr'], loss.item())
    train_loss = running_loss / step
    train_accuracy_t = accuracy_t / train_batch_num
    train_accuracy_s = accuracy_s / train_batch_num
    print('teacher model train acc is {:.5f}, students model train acc is {:.5f}'.format(train_accuracy_t, train_accuracy_s))

    # 每训练一轮就进行一次评估
    val_accuracy_t, val_accuracy_s = validate(val_loader, model, device)
    print('teacher model val acc is {:.5f}, students model val acc is {:.5f}'.format(val_accuracy_t, val_accuracy_s))

    # 保存日志文件以及last和best文件模型
    save_path = args.save_path + '{}.txt'.format(args.tag)
    message = 'train epoch[{}/{}] lr:{} loss:{:.3f}'.format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr'], train_loss) + '\n' + \
              'teacher train accurate:{:.5f}  teacher val accurate:{:.5f}, student train accurate:{:.5f}  student val accurate:{:.5f}'.format\
                  (train_accuracy_t, val_accuracy_t, train_accuracy_s, val_accuracy_s)
    writefile(save_path, message, 'a+')
    checkpoint_save_path = args.save_path + '{}_last.pth'.format(args.tag)
    checkpoint_save(model, optimizer, checkpoint_save_path)
    if val_accuracy_s >= best_acc:
        best_acc = val_accuracy_s
        best_save_path = args.save_path + '{}_best.txt'.format(args.tag)
        writefile(best_save_path, message, 'w')
        checkpoint_save_path = args.save_path + '{}_best.pth'.format(args.tag)
        checkpoint_save(model, optimizer, checkpoint_save_path)
    return best_acc


if __name__ == '__main__':
    main()
