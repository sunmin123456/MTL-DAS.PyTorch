import datetime
import glob
import io
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from ptflops import get_model_complexity_info
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from thop import profile
from torch import optim
from torch.autograd import Variable
import torch
import os
# 训练/验证过程
import torch.nn as nn
import torch.nn.functional as F
from torchstat import stat
# 适用于敲击3段划分
from data_pre_algorithm import draw_confusion_matrix
from 多任务学习原始数据准备 import Dataset_mat_MTL
from shutil import copyfile


# 定义残差块ResBlock


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=(1, 1), stride=(stride, stride), bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


# resnet里面的
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=(stride, stride), bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride),
                     padding=(dilation, dilation), groups=groups, bias=False,
                     dilation=(dilation, dilation))


# resnet里面的，后面需要优化, 没用到了
class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 改成单任务网络
# 需要修改的位置：
# 1.self.tasks
# 2.最后的输出
# 若只修改网络，则训练函数不需要变

class Simple_MTL_Net(nn.Module):
    def __init__(self):
        super(Simple_MTL_Net, self).__init__()

        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.BN1 = nn.BatchNorm2d(16)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        self.cnn1 = nn.Sequential(self.Conv1, self.BN1, self.relu1, self.pool1)

        self.Conv2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.BN2 = nn.BatchNorm2d(32)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        self.cnn2 = nn.Sequential(self.Conv2, self.BN2, self.relu2, self.pool2)

        self.task1pool = nn.AdaptiveAvgPool2d((1, 1))
        self.task2pool = nn.AdaptiveAvgPool2d((1, 1))

        self.task1pool1d = nn.AvgPool1d(kernel_size=self.ch[-1] // 16, stride=self.ch[-1] // 16)
        self.task2pool1d = nn.AvgPool1d(kernel_size=self.ch[-1] // 2, stride=self.ch[-1] // 2)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)

        pred1 = self.task1pool(x).squeeze(2).squeeze(2)
        pred2 = self.task2pool(x).squeeze(2).squeeze(2)

        pred1 = self.task1pool1d(pred1.unsqueeze(1)).squeeze(1)
        pred2 = self.task2pool1d(pred2.unsqueeze(1)).squeeze(1)

        pred1 = F.log_softmax(pred1, dim=1)
        pred2 = F.log_softmax(pred2, dim=1)
        return pred1, pred2


class ResNet1(nn.Module):
    def __init__(self, ResBlock=ResBlock, ch=16, k_size=7, stride=3, res_num=8, t='multi'):
        super(ResNet1, self).__init__()

        self.res_num = res_num

        self.t = t

        self.ch = [ch, ch]

        for i in range(self.res_num // 2 - 1):
            self.ch.append(ch * (2 ** (i + 1)))

        if self.t == 'multi':

            self.tasks = ['distance', 'event']
        else:
            self.tasks = ['single']

        # ResNet层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.ch[0], kernel_size=(k_size, k_size), stride=(stride, stride), padding=(2, 2), bias=False),
            nn.BatchNorm2d(self.ch[0]),
            nn.ReLU()
        )

        self.resblock1 = ResBlock(inchannel=self.ch[0], outchannel=self.ch[1], stride=1)
        self.resblock2 = ResBlock(inchannel=self.ch[1], outchannel=self.ch[1], stride=1)
        self.resblock3 = ResBlock(inchannel=self.ch[1], outchannel=self.ch[2], stride=2)
        self.resblock4 = ResBlock(inchannel=self.ch[2], outchannel=self.ch[2], stride=1)

        if self.res_num > 4:
            self.resblock5 = ResBlock(inchannel=self.ch[2], outchannel=self.ch[3], stride=2)
            self.resblock6 = ResBlock(inchannel=self.ch[3], outchannel=self.ch[3], stride=1)
        if self.res_num > 6:
            self.resblock7 = ResBlock(inchannel=self.ch[3], outchannel=self.ch[4], stride=2)
            self.resblock8 = ResBlock(inchannel=self.ch[4], outchannel=self.ch[4], stride=1)
        if self.res_num > 8:
            self.resblock9 = ResBlock(inchannel=self.ch[4], outchannel=self.ch[5], stride=2)
            self.resblock10 = ResBlock(inchannel=self.ch[5], outchannel=self.ch[5], stride=1)

        # 注意力层
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.encoder_att_1 = nn.ModuleList(
            [self.att_layer(self.ch[1], self.ch[1] // 2, self.ch[1]) for _ in self.tasks])  # 生成任务数量个注意力层

        self.encoder_att_2 = nn.ModuleList(
            [self.att_layer(2 * self.ch[2], self.ch[2] // 2, self.ch[2]) for _ in self.tasks])

        if self.res_num > 4:
            self.encoder_att_3 = nn.ModuleList(
                [self.att_layer(2 * self.ch[3], self.ch[3] // 2, self.ch[3]) for _ in self.tasks])

        if self.res_num > 6:
            self.encoder_att_4 = nn.ModuleList(
                [self.att_layer(2 * self.ch[4], self.ch[4] // 2, self.ch[4]) for _ in self.tasks])

        if self.res_num > 8:
            self.encoder_att_5 = nn.ModuleList(
                [self.att_layer(2 * self.ch[5], self.ch[5] // 2, self.ch[5]) for _ in self.tasks])

        # Define task shared attention encoders using residual bottleneck layers
        # We do not apply shared attention encoders at the last layer,
        # so the attended features will be directly fed into the task-specific decoders.

        # 修改解码器层 12.22 这里没有用到bottleneck，只是简单的1x1卷积

        self.encoder_block_att_1 = nn.ModuleList(
            [nn.Sequential(conv3x3(self.ch[1], self.ch[2], stride=1), nn.BatchNorm2d(self.ch[2]), nn.ReLU(inplace=True))
             for _ in self.tasks])

        if self.res_num > 4:
            self.encoder_block_att_2 = nn.ModuleList(
                [nn.Sequential(conv3x3(self.ch[2], self.ch[3], stride=1), nn.BatchNorm2d(self.ch[3]),
                               nn.ReLU(inplace=True)) for _ in
                 self.tasks])

        if self.res_num > 6:
            self.encoder_block_att_3 = nn.ModuleList(
                [nn.Sequential(conv3x3(self.ch[3], self.ch[4], stride=1), nn.BatchNorm2d(self.ch[4]),
                               nn.ReLU(inplace=True)) for _ in
                 self.tasks])

        if self.res_num > 8:
            self.encoder_block_att_4 = nn.ModuleList(
                [nn.Sequential(conv3x3(self.ch[4], self.ch[5], stride=1), nn.BatchNorm2d(self.ch[5]),
                               nn.ReLU(inplace=True)) for _ in
                 self.tasks])

        # 输出层
        self.task1pool = nn.AdaptiveAvgPool2d((1, 1))
        self.task2pool = nn.AdaptiveAvgPool2d((1, 1))
        self.task3pool = nn.AdaptiveAvgPool2d((1, 1))

        self.task1pool1d = nn.AvgPool1d(kernel_size=self.ch[-1] // 16, stride=self.ch[-1] // 16)
        self.task2pool1d = nn.AvgPool1d(kernel_size=self.ch[-1] // 2, stride=self.ch[-1] // 2)
        self.task3pool1d = nn.AvgPool1d(kernel_size=self.ch[-1] // 32, stride=self.ch[-1] // 32)

        # self.output_layer=nn.Sequential(
        #     nn.Conv3d(in_channels=self.ch[4], out_channels=self.ch[4], kernel_size=(3, 3, 3)),  # no padding
        #     nn.BatchNorm3d(self.ch[4]),
        #     nn.ReLU())

    def forward(self, x):
        # 在这里，整个ResNet18的结构就很清晰了
        x = self.conv1(x)

        ubt = []  # 通道数：16，16，32，32，64，64，128，128

        # 得到8个中间输出，同时让网络正向传播
        ubt.append(self.resblock1(x))
        ubt.append(self.resblock2(ubt[-1]))
        ubt.append(self.resblock3(ubt[-1]))
        ubt.append(self.resblock4(ubt[-1]))

        if self.res_num > 4:
            ubt.append(self.resblock5(ubt[-1]))
            ubt.append(self.resblock6(ubt[-1]))

        if self.res_num > 6:
            ubt.append(self.resblock7(ubt[-1]))
            ubt.append(self.resblock8(ubt[-1]))

        if self.res_num > 8:
            ubt.append(self.resblock9(ubt[-1]))
            ubt.append(self.resblock10(ubt[-1]))

        # 注意力层

        a_1_mask = [att_i(ubt[0]) for att_i in self.encoder_att_1]  # Generate task specific attention map
        a_1 = [a_1_mask_i * ubt[1] for a_1_mask_i in a_1_mask]  # Apply task specific attention map to shared features
        a_1 = [self.down_sampling(encoder_block(a_1_i)) for a_1_i, encoder_block in
               zip(a_1, self.encoder_block_att_1)]  # 32,50,192

        if self.res_num == 4:

            a_5_mask = [att_i(torch.cat((ubt[2], a_3_i), dim=1)) for a_3_i, att_i in zip(a_1, self.encoder_att_2)]
            a_5 = [a_4_mask_i * ubt[3] for a_4_mask_i in a_5_mask]  # a_4是注意力网络的最终输出 128,13,48


        elif self.res_num == 6:

            a_2_mask = [att_i(torch.cat((ubt[2], a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)]
            a_2 = [a_2_mask_i * ubt[3] for a_2_mask_i in a_2_mask]
            a_2 = [self.down_sampling(encoder_block(a_2_i)) for a_2_i, encoder_block in
                   zip(a_2, self.encoder_block_att_2)]  # 32,50,192

            a_5_mask = [att_i(torch.cat((ubt[4], a_3_i), dim=1)) for a_3_i, att_i in zip(a_2, self.encoder_att_3)]
            a_5 = [a_4_mask_i * ubt[5] for a_4_mask_i in a_5_mask]  # a_4是注意力网络的最终输出 128,13,48


        elif self.res_num == 8:

            a_2_mask = [att_i(torch.cat((ubt[2], a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)]
            a_2 = [a_2_mask_i * ubt[3] for a_2_mask_i in a_2_mask]
            a_2 = [self.down_sampling(encoder_block(a_2_i)) for a_2_i, encoder_block in
                   zip(a_2, self.encoder_block_att_2)]  # 32,50,192

            a_3_mask = [att_i(torch.cat((ubt[4], a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
            a_3 = [a_3_mask_i * ubt[5] for a_3_mask_i in a_3_mask]
            a_3 = [self.down_sampling(encoder_block(a_3_i)) for a_3_i, encoder_block in
                   zip(a_3, self.encoder_block_att_3)]  # 32,50,192

            a_5_mask = [att_i(torch.cat((ubt[6], a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_att_4)]
            a_5 = [a_4_mask_i * ubt[7] for a_4_mask_i in a_5_mask]  # a_4是注意力网络的最终输出 128,13,48

        else:

            a_2_mask = [att_i(torch.cat((ubt[2], a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)]
            a_2 = [a_2_mask_i * ubt[3] for a_2_mask_i in a_2_mask]
            a_2 = [self.down_sampling(encoder_block(a_2_i)) for a_2_i, encoder_block in
                   zip(a_2, self.encoder_block_att_2)]  # 32,50,192

            a_3_mask = [att_i(torch.cat((ubt[4], a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
            a_3 = [a_3_mask_i * ubt[5] for a_3_mask_i in a_3_mask]
            a_3 = [self.down_sampling(encoder_block(a_3_i)) for a_3_i, encoder_block in
                   zip(a_3, self.encoder_block_att_3)]  # 32,50,192

            a_4_mask = [att_i(torch.cat((ubt[6], a_2_i), dim=1)) for a_2_i, att_i in zip(a_3, self.encoder_att_4)]
            a_4 = [a_3_mask_i * ubt[7] for a_3_mask_i in a_4_mask]
            a_4 = [self.down_sampling(encoder_block(a_3_i)) for a_3_i, encoder_block in
                   zip(a_4, self.encoder_block_att_4)]  # 32,50,192

            a_5_mask = [att_i(torch.cat((ubt[8], a_3_i), dim=1)) for a_3_i, att_i in zip(a_4, self.encoder_att_5)]
            a_5 = [a_4_mask_i * ubt[9] for a_4_mask_i in a_5_mask]  # a_4是注意力网络的最终输出 128,13,48

        # 输出层

        if self.t == 'multi':

            pred1 = self.task1pool(a_5[0]).squeeze(2).squeeze(2)
            pred2 = self.task2pool(a_5[1]).squeeze(2).squeeze(2)

            pred1 = self.task1pool1d(pred1.unsqueeze(1)).squeeze(1)
            pred2 = self.task2pool1d(pred2.unsqueeze(1)).squeeze(1)

            pred1 = F.log_softmax(pred1, dim=1)
            pred2 = F.log_softmax(pred2, dim=1)

            return pred1, pred2
        elif self.t == 'event':
            pred1 = self.task2pool(a_5[0]).squeeze(2).squeeze(2)
            pred1 = self.task2pool1d(pred1.unsqueeze(1)).squeeze(1)
            pred1 = F.log_softmax(pred1, dim=1)
            return pred1

        elif self.t == 'distance':
            pred1 = self.task1pool(a_5[0]).squeeze(2).squeeze(2)
            pred1 = self.task1pool1d(pred1.unsqueeze(1)).squeeze(1)
            pred1 = F.log_softmax(pred1, dim=1)
            return pred1

        else:
            pred1 = self.task3pool(a_5[0]).squeeze(2).squeeze(2)
            pred1 = self.task3pool1d(pred1.unsqueeze(1)).squeeze(1)
            pred1 = F.log_softmax(pred1, dim=1)
            return pred1

    def att_layer(self, in_channel, intermediate_channel, out_channel):  # 输入通道，中间通道，输出通道
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=(1, 1)),  # 只用于降维
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid())


# 训练/验证过程
def train_multi(model, epoch_num, start_epoch, optimizer, criterion, data_loader, save_dir,
                confusion=True, use_gpu=True, save_output=False, dwa=False, T=2.0, is_test=False):
    start_time = datetime.datetime.now()

    # 读取学习率参数，将学习率除以5
    def adjust_lr(optimizer):
        for param in optimizer.param_groups:
            param["lr"] = param["lr"] / 1.5
        return optimizer

    LossLine = [[], []]
    AccLine = [[], []]
    testAccLine = [[], []]
    testLossLine = [[], []]
    acc_sum100 = [0, 0]
    loss_sum100 = [0, 0]

    # -与dwa有关的参数
    lambda_weight = np.ones([2, epoch_num])  # 为任务数量,epoch数
    avg_cost = np.zeros([epoch_num, 2], dtype=np.float32)  # 为epoch数,任务数量，表示不同epoch下不同任务的平均损失值
    # -与dwa有关的参数

    # LossLine = np.load('LossLine.npy')
    # AccLine =np.load('AccLine.npy')
    # testAccLine = np.load('testAccLine.npy')

    for epoch in range(start_epoch, epoch_num):

        # 每5个epoch测试一次，绘制混淆矩阵，并保存模型
        if epoch % 5 == 0:

            # if epoch < 20:
            optimizer = adjust_lr(optimizer)

            model.train(False)
            model.eval()
            ValAcc = [0, 0]
            ValBatch = [0, 0]
            testloss = [0, 0]

            pred_lst = [[], []]
            label_lst = [[], []]

            # t_predict = datetime.datetime.now()
            for val_cnt, val_data in enumerate(data_loader["val"]):

                valX, distance_label, event_label = val_data

                ######################################
                if use_gpu:
                    valX = Variable(valX.cuda())
                    distance_label = Variable(distance_label.cuda())
                    event_label = Variable(event_label.cuda())
                else:
                    valX = Variable(valX)
                    distance_label = Variable(distance_label)
                    event_label = Variable(event_label)
                ######################################

                # 模型输出
                out1, out2 = model(valX)

                l1 = criterion(out1, distance_label.long())
                l2 = criterion(out2, event_label.long())

                testloss[0] += l1.cpu().data  # 一个batch的损失值
                testloss[1] += l2.cpu().data

                pred1 = torch.max(out1, 1)[1]
                pred2 = torch.max(out2, 1)[1]

                # print('模型运行时间：{}'.format(datetime.datetime.now() - t_predict))

                ValAcc[0] += (pred1 == distance_label).sum()
                ValBatch[0] += len(pred1)

                ValAcc[1] += (pred2 == event_label).sum()
                ValBatch[1] += len(pred2)

                label_lst[0].extend(distance_label)  # 添加数据
                pred_lst[0].extend(pred1)  # 添加数据
                label_lst[1].extend(event_label)  # 添加数据
                pred_lst[1].extend(pred2)  # 添加数据

            # print('总推理时间：{}'.format(datetime.datetime.now() - t_predict))
            # return

            acc1 = float(ValAcc[0]) / ValBatch[0]
            acc2 = float(ValAcc[1]) / ValBatch[1]
            testLossLine[0].append(
                testloss[0] / len(data_loader['val'].dataset))  # 有问题！！！应当除以总样本数len(data_loader['val'])
            testLossLine[1].append(testloss[1] / len(data_loader['val'].dataset))

            testAccLine[0] = np.append(testAccLine[0], acc1)
            testAccLine[1] = np.append(testAccLine[1], acc2)
            print("{}\nepoch:{}  正确率: distance:{}  event:{}".format("*" * 50, epoch, acc1, acc2))

            label_lst[0] = [int(x) for x in label_lst[0]]
            pred_lst[0] = [int(x) for x in pred_lst[0]]
            label_lst[1] = [int(x) for x in label_lst[1]]
            pred_lst[1] = [int(x) for x in pred_lst[1]]

            con_mat = [confusion_matrix(label_lst[0], pred_lst[0]), confusion_matrix(label_lst[1], pred_lst[1])]

            for taskindex, taskname in enumerate(['任务1：distance', '任务2：event']):
                print('任务1：{}'.format(taskname))
                print(con_mat[taskindex])
                print('准确率：{}'.format(accuracy_score(label_lst[taskindex], pred_lst[taskindex])))
                print(f1_score(label_lst[taskindex], pred_lst[taskindex], average=None))
                print('F1_Score：{}'.format(f1_score(label_lst[taskindex], pred_lst[taskindex], average='weighted')))
                print('精确率：{}'.format(
                    precision_score(label_lst[taskindex], pred_lst[taskindex], average='weighted')))
                print('召回率：{}'.format(recall_score(label_lst[taskindex], pred_lst[taskindex], average='weighted')))

            if save_output:
                if not is_test:
                    np.save(save_dir + '/testAccLine', testAccLine)
                    np.save(save_dir + '/testLossLine', testLossLine)

                if acc1 >= 0.98:
                    torch.save(model.state_dict(), os.path.join(save_dir,
                                                                "{}__{:.5f}_{}.pth".format(
                                                                    datetime.datetime.now().strftime(
                                                                        "%Y_%m_%d__%H_%M_%S"),
                                                                    acc1, epoch)))
                    # torch.save(model.state_dict(), PATH)
                    np.save(save_dir + '/confusion matrix distance {:.5f} {}.npy'.format(acc1, epoch), con_mat[0])
                    np.save(save_dir + '/confusion matrix event {:.5f} {}.npy'.format(acc2, epoch), con_mat[1])

            if is_test:
                return

        if epoch < epoch_num - 1:
            if dwa:
                if epoch == 0 or epoch == 1:
                    lambda_weight[:, epoch] = 1.0
                else:
                    w_1 = avg_cost[epoch - 1, 0] / avg_cost[epoch - 2, 0]
                    w_2 = avg_cost[epoch - 1, 1] / avg_cost[epoch - 2, 1]

                    # 以下两个参数为论文中的λ
                    lambda_weight[0, epoch] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
                    lambda_weight[1, epoch] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
                    print('lambda_weight:')
                    print(lambda_weight)
                    # print(avg_cost)

            model.train(True)

            for batch_cnt, data in enumerate(data_loader['train']):

                bacthX, distance_label, event_label = data

                ######################################
                if use_gpu:
                    bacthX = Variable(bacthX.cuda())
                    distance_label = Variable(distance_label.cuda())
                    event_label = Variable(event_label.cuda())
                else:
                    bacthX = Variable(bacthX)
                    distance_label = Variable(distance_label)
                    event_label = Variable(event_label)
                ######################################
                out1, out2 = model(bacthX)

                # -########处理损失值############

                loss = [criterion(out1, distance_label.long()), criterion(out2, event_label.long())]

                avg_cost[epoch, 0] += loss[0].data
                avg_cost[epoch, 1] += loss[1].data

                if dwa:
                    loss_balance = sum([lambda_weight[i, epoch] * loss[i] for i in range(2)])
                else:
                    loss_balance = loss[0] + loss[1]

                # -########处理损失值############

                pred1 = torch.max(out1, 1)[1]
                pred2 = torch.max(out2, 1)[1]

                optimizer.zero_grad()
                loss_balance.backward()
                optimizer.step()

                # 做100次求和
                acc_sum100[0] += float(torch.sum((pred1 == distance_label)).data) / len(pred1)
                acc_sum100[1] += float(torch.sum((pred2 == event_label)).data) / len(pred2)

                loss_sum100[0] += loss[0].cpu().data.item() / len(pred1)
                loss_sum100[1] += loss[1].cpu().data.item() / len(pred2)

                # 每训练一定的batch就输出一次精确度，该精确度为本轮epoch已训练部分的值,同时保存曲线
                if (batch_cnt + 1) % 100 == 0:
                    acc_sum100[0] = acc_sum100[0] / 100
                    loss_sum100[0] = loss_sum100[0] / 100
                    acc_sum100[1] = acc_sum100[1] / 100
                    loss_sum100[1] = loss_sum100[1] / 100

                    print("epoch-iteration:{}-{}, loss:{}, accuracy:{}".format(epoch + 1, (batch_cnt + 1), loss_sum100,
                                                                               acc_sum100))
                    print("time:{}".format(datetime.datetime.now() - start_time))

                    LossLine = [np.append(LossLine[0], loss_sum100[0]), np.append(LossLine[1], loss_sum100[1])]
                    AccLine = [np.append(AccLine[0], acc_sum100[0]), np.append(AccLine[1], acc_sum100[1])]
                    if save_output:
                        np.save(save_dir + '/trainLossLine', LossLine)
                        np.save(save_dir + '/trainAccLine', AccLine)
                    acc_sum100 = [0, 0]
                    loss_sum100 = [0, 0]

            print("第{}个epoch训练完成！".format(epoch + 1))
            avg_cost[epoch, :] /= len(data_loader['train'])

            acc_sum100 = [0, 0]
            loss_sum100 = [0, 0]


# 统计模型需要训练的参数量
def count_parameters(model, grad=True):
    if grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


class Logger(object):
    def __init__(self, filename="Default.log", path="./"):
        # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        self.terminal = sys.stdout
        self.log = ''
        self.filename = filename
        self.path = path

    def write(self, message):
        self.terminal.write(message)
        self.log += message

    def save(self):
        with open(self.path + '/' + self.filename, 'a') as f:
            f.write(self.log)

    def flush(self):
        pass


def main(ch=16, k_size=7, stride=3, res_num=8, task='multi', is_test=False, pth_file='', random_state=1,fold_index=1):
    if task == 'multi':

        note = '多任务 task={} ch={} k_size={} stride={} res_num={} is_test={}'.format(task, ch, k_size, stride, res_num,
                                                                                    is_test)
    elif task == 'distance':
        note = '单任务垂距 task={} ch={} k_size={} stride={} res_num={} is_test={}'.format(task, ch, k_size, stride, res_num,
                                                                                      is_test)
    else:
        note = '单任务事件 task={} ch={} k_size={} stride={} res_num={} is_test={}'.format(task, ch, k_size, stride, res_num,
                                                                                      is_test)

    save_dir = r'\\121.48.161.226\kb208datapool\LabFiles\users\wyf\保存结果\{}{}/'.format(
        datetime.datetime.now().strftime("%m月%d日%H_%M_%S"), note)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir + 'src'):
        os.makedirs(save_dir + 'src')

    py_file = glob.glob(r'./*.py')
    for filename in py_file:
        copyfile(filename, save_dir + 'src/' + os.path.basename(filename))

    log1 = Logger('训练过程.log', path=save_dir)
    sys.stdout = log1
    print(__file__)

    GPU_device = True
    if not torch.cuda.is_available():
        GPU_device = False

    batchsize = 32
    numworks = 0
    model = ResNet1(ch=ch, k_size=k_size, stride=stride, res_num=res_num, t=task)
    if not pth_file == '':
        model.load_state_dict(torch.load(pth_file, map_location='cpu'), strict=True)

    # 和标准的ResNet118对比
    print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(model),
                                                             count_parameters(model) / 11689512))
    print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(model, grad=False),
                                                             count_parameters(model, grad=False) / 11689512))
    # -计算模型复杂度
    # macs, params = profile(model, inputs=(torch.randn((1,1,40,50,63)),))
    # print('{:<30}  {:<8}'.format('MACs: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    macs, params = get_model_complexity_info(model, (1, 100, 250), as_strings=False,
                                             print_per_layer_stat=False, verbose=True, ost=log1)
    print('{:<30}  {:<8}'.format('MACs: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # -计算模型复杂度

    if GPU_device == True:
        model.cuda()

    base_lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.00001)
    criterion = torch.nn.NLLLoss()  # criterion = torch.nn.CrossEntropyLoss()

    if is_test:
        dataset_dir0324 = r'E:\dataset\原始时空矩阵降采样增广0324qiaoji分训练测试/test/'
        dataset_dirwajue = r'E:\dataset\原始时空矩阵降采样增广wajue分训练测试/test/'
    else:
        dataset_dir0324 = r'E:\dataset\原始时空矩阵降采样增广0324qiaoji分训练测试/train/'
        dataset_dirwajue = r'E:\dataset\原始时空矩阵降采样增广wajue分训练测试/train/'


    dataset_MTL = Dataset_mat_MTL(dataset_dir0324=dataset_dir0324, dataset_dirwajue=dataset_dirwajue, ram=True,
                                  is_test=is_test, random_state=random_state, fold_index=fold_index)

    dataloader1 = {}
    dataloader1['train'] = torch.utils.data.DataLoader(dataset_MTL.dataset['train'], batch_size=batchsize,
                                                       shuffle=True, num_workers=numworks)
    dataloader1['val'] = torch.utils.data.DataLoader(dataset_MTL.dataset['val'], batch_size=batchsize,
                                                     shuffle=True, num_workers=numworks)

    # # 保存样本名-标签列表
    # dataset_MTL.dataset['train'].get_name_label_csv(savedir='train name label.csv')
    # dataset_MTL.dataset['val'].get_name_label_csv(savedir='val name label.csv')

    start_time = datetime.datetime.now()
    print('数据集准备完毕，开始训练：{}'.format(start_time))
    is_train = True
    if is_train:

        if task == 'multi':
            train_multi(model=model, data_loader=dataloader1, epoch_num=41, start_epoch=0, optimizer=optimizer,
                        criterion=criterion, use_gpu=GPU_device, save_dir=save_dir, save_output=True, is_test=is_test)
        # elif task == 'distance':
        #     train_distance(model=model, data_loader=dataloader1, epoch_num=41, start_epoch=0, optimizer=optimizer,
        #                    criterion=criterion, use_gpu=GPU_device, save_dir=save_dir, save_output=True,
        #                    is_test=is_test)
        # else:
        #     train_event(model=model, data_loader=dataloader1, epoch_num=41, start_epoch=0, optimizer=optimizer,
        #                 criterion=criterion, use_gpu=GPU_device, save_dir=save_dir, save_output=True, is_test=is_test)

    # -绘制四种曲线并保存

    if not is_test:

        linelist = ['trainAccLine', 'trainLossLine',
                    'testAccLine', 'testLossLine']
        for linename in linelist:
            line = np.load(save_dir + linename + '.npy')
            plt.figure()
            plt.plot(line[0], label='distance')
            plt.plot(line[1], label='event')
            plt.legend()
            plt.savefig(save_dir + linename + '.png')
            plt.close()





    else:
        leibie_event = ['Striking', 'Excavating ']
        leibei_distance = ['{}m'.format(i) for i in range(16)]
        cm_list = glob.glob(save_dir + 'confusion matrix*.npy')
        for cm in cm_list:
            mat1 = np.load(cm)
            if len(mat1[0]) == 16:
                draw_confusion_matrix(confusion_matrix=mat1, leibie1=leibei_distance, figsize=(6.5, 6),
                                      savepath=save_dir + '/confusion matrix distance.svg')
            elif len(mat1[0]) == 2:
                draw_confusion_matrix(confusion_matrix=mat1, leibie1=leibie_event, figsize=(2.8, 2.8),
                                      savepath=save_dir + '/confusion matrix event.svg')
            else:
                raise ValueError

    # -绘制四种曲线并保存
    log1.save()

    if not is_test:
        line = np.load(save_dir + 'testAccLine.npy')
        with open('./' + 'accuracylist.txt', 'a') as f:
            f.write('\n' + note + '\n')
            f.write(str(np.max(line, axis=1)[0]))
            f.write('\n')
            f.write(str(np.max(line, axis=1)[1]))


if __name__ == '__main__':
    main()
