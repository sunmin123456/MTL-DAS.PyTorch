import datetime
import warnings
from shutil import copyfile

import torch.utils.data
import torch
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from torchvision.models import inception_v3, InceptionOutputs
import torch.nn as nn
from torchvision.models.inception import InceptionA, InceptionB, InceptionC, InceptionD, InceptionAux, InceptionE
from torch import optim

from data_pre_algorithm import draw_confusion_matrix
from 多任务学习原始数据准备 import Dataset_mat_MTL
from torch.autograd import Variable
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import glob
import sys


class BasicConv2d(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            **kwargs
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception3(nn.Module):

    def __init__(
            self,
            num_classes: int = 32,
            aux_logits: bool = False,
            transform_input: bool = False,
            inception_blocks=None,
            init_weights=True
    ) -> None:
        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        if init_weights is None:
            warnings.warn('The default weight initialization of inception_v3 will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        # 这里的结构和后面forward一样

        '''
        模型优化策略
        池化改成avg,正确率反而降低
        更改学习率



        '''

        # 原始参数设置
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(1, 32, kernel_size=3, stride=2)  # 将这里改成1
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.AuxLogits = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, num_classes)

        # self.aux_logits = aux_logits
        # self.transform_input = transform_input
        # self.Conv2d_1a_3x3 = conv_block(1, 64, kernel_size=3, stride=2)  # 将这里改成1
        # self.Conv2d_2a_3x3 = conv_block(64, 64, kernel_size=3)
        # self.Conv2d_2b_3x3 = conv_block(64, 128, kernel_size=3, padding=1)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # self.Conv2d_3b_1x1 = conv_block(128, 160, kernel_size=1)
        # self.Conv2d_4a_3x3 = conv_block(160, 384, kernel_size=3)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # self.Mixed_5b = inception_a(384, pool_features=64)
        # self.Mixed_5c = inception_a(288, pool_features=128)
        # self.Mixed_5d = inception_a(352, pool_features=128)
        # self.Mixed_6a = inception_b(352)
        # self.Mixed_6b = inception_c(832, channels_7x7=128)
        # self.Mixed_6c = inception_c(768, channels_7x7=160)
        # self.Mixed_6d = inception_c(768, channels_7x7=160)
        # self.Mixed_6e = inception_c(768, channels_7x7=192)
        # self.AuxLogits = None
        # if aux_logits:
        #     self.AuxLogits = inception_aux(768, num_classes)
        # self.Mixed_7a = inception_d(768)
        # self.Mixed_7b = inception_e(1280)
        # self.Mixed_7c = inception_e(2048)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout()
        # self.fc = nn.Linear(2048, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x, aux):
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x):
        # x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)


# 训练/验证过程
def train(model, epoch_num, start_epoch, optimizer, criterion, data_loader, save_dir,
          confusion=True, use_gpu=True, save_output=False,is_test=False):
    hash_list = [[i % 16, i // 16] for i in range(32)]
    start_time = datetime.datetime.now()
    print('开始训练：{}'.format(start_time))

    # 读取学习率参数，将学习率除以5
    def adjust_lr(optimizer):
        for param in optimizer.param_groups:
            param["lr"] = param["lr"] / 1.5
        return optimizer

    LossLine = [[], []]
    AccLine = [[], []]
    testAccLine = [[], []]
    testLossLine = [[], []]  #
    acc_sum100 = [0, 0]
    loss_sum100 = [0, 0]

    # LossLine = np.load('LossLine.npy')
    # AccLine =np.load('AccLine.npy')
    # testAccLine = np.load('testAccLine.npy')

    for epoch in range(start_epoch, epoch_num):

        # 每5个epoch测试一次，绘制混淆矩阵，并保存模型
        if epoch % 5 == 0:  # or epoch ==0:

            # if epoch < 20:
            if epoch != 0:
                optimizer = adjust_lr(optimizer)

            model.train(False)
            model.eval()
            ValAcc = [0, 0]
            ValBatch = [0, 0]
            testloss = [0, 0]  #
            pred_lst = [[], []]
            label_lst = [[], []]

            # t_predict = datetime.datetime.now()
            for val_cnt, val_data in enumerate(data_loader["val"]):
                # t_predict=datetime.datetime.now()
                valX, mix_label = val_data

                ######################################
                if use_gpu:
                    valX = Variable(valX.cuda())
                    mix_label = Variable(mix_label.cuda())
                    # event_label = Variable(event_label.cuda())
                else:
                    valX = Variable(valX)
                    mix_label = Variable(mix_label)
                    # event_label = Variable(event_label)
                ######################################

                # 模型输出

                out1 = model(valX)
                out1_pred = torch.max(out1, 1)[1]
                l1 = criterion(out1, mix_label.long())  #

                testloss[0] += l1.cpu().data  #
                testloss[1] += l1.cpu().data  #

                # print('模型运行时间：{}'.format(datetime.datetime.now() - t_predict))
                # -进行标签变换
                pred1 = torch.zeros(out1_pred.shape[0])
                pred2 = torch.zeros(out1_pred.shape[0])
                for switch_i in range(out1_pred.shape[0]):
                    pred1[switch_i] = hash_list[out1_pred[switch_i].detach().cpu().item()][0]
                    pred2[switch_i] = hash_list[out1_pred[switch_i].detach().cpu().item()][1]

                distance_label = torch.zeros(mix_label.shape[0])
                event_label = torch.zeros(mix_label.shape[0])
                for switch_i in range(mix_label.shape[0]):
                    distance_label[switch_i] = hash_list[mix_label[switch_i].detach().cpu().item()][0]
                    event_label[switch_i] = hash_list[mix_label[switch_i].detach().cpu().item()][1]
                # -进行标签变换
                # print('总推理时间：{}'.format(datetime.datetime.now() - t_predict))

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
                testloss[0] / len(data_loader['val'].dataset))  # # 有问题！！！应当除以总样本数len(data_loader['val'])
            testLossLine[1].append(testloss[1] / len(data_loader['val'].dataset))  #

            testAccLine[0] = np.append(testAccLine[0], acc1)
            testAccLine[1] = np.append(testAccLine[1], acc2)
            print("{}\nepoch:{}  正确率: distance:{}  event:{}".format("*" * 50, epoch + 1, acc1, acc2))

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
                    np.save(save_dir + '/testLossLine', testLossLine)  #

                if acc1 >= 0.95:
                    torch.save(model.state_dict(), os.path.join(save_dir,
                                                                "{}__{:.5f}_{}.pth".format(
                                                                    datetime.datetime.now().strftime(
                                                                        "%Y_%m_%d__%H_%M_%S"),
                                                                    acc1, epoch)))
                    # torch.save(model.state_dict(), PATH)
                    np.save(save_dir + '/confusion matrix {} {}.npy'.format(acc1, epoch), con_mat[0])  #
                    np.save(save_dir + '/confusion matrix {} {}.npy'.format(acc2, epoch), con_mat[1])  #
            if is_test:
                return

        if epoch < epoch_num - 1:

            model.train(True)

            for batch_cnt, data in enumerate(data_loader['train']):

                bacthX, mix_label = data

                ######################################
                if use_gpu:
                    bacthX = Variable(bacthX.cuda())
                    mix_label = Variable(mix_label.cuda())
                    # event_label = Variable(event_label.cuda())
                else:
                    bacthX = Variable(bacthX)
                    mix_label = Variable(mix_label)
                    # event_label = Variable(event_label)
                ######################################
                out1 = model(bacthX)

                # -########处理损失值############
                loss = [criterion(out1, mix_label.long())]
                loss_balance = loss[0]

                # -########处理损失值############
                optimizer.zero_grad()
                loss_balance.backward()
                optimizer.step()

                out1_pred = torch.max(out1, 1)[1]
                # -标签转换
                pred1 = torch.zeros(out1_pred.shape[0])
                pred2 = torch.zeros(out1_pred.shape[0])
                for switch_i in range(out1_pred.shape[0]):
                    pred1[switch_i] = hash_list[out1_pred[switch_i].detach().cpu().item()][0]
                    pred2[switch_i] = hash_list[out1_pred[switch_i].detach().cpu().item()][1]

                distance_label = torch.zeros(mix_label.shape[0])
                event_label = torch.zeros(mix_label.shape[0])
                for switch_i in range(mix_label.shape[0]):
                    distance_label[switch_i] = hash_list[mix_label[switch_i].detach().cpu().item()][0]
                    event_label[switch_i] = hash_list[mix_label[switch_i].detach().cpu().item()][1]
                # -标签转换

                # 做100次求和
                acc_sum100[0] += float(torch.sum((pred1 == distance_label)).data) / len(pred1)
                acc_sum100[1] += float(torch.sum((pred2 == event_label)).data) / len(pred2)

                loss_sum100[0] += loss[0].cpu().data.item()/ len(out1_pred)
                loss_sum100[1] += loss[0].cpu().data.item()/ len(out1_pred)

                # 每训练一定的batch就输出一次精确度，该精确度为本轮epoch已训练部分的值,同时保存曲线
                if (batch_cnt + 1) % 100 == 0:
                    acc_sum100[0] = acc_sum100[0] / 100
                    loss_sum100[0] = loss_sum100[0] / 100
                    acc_sum100[1] = acc_sum100[1] / 100
                    loss_sum100[1] = loss_sum100[1] / 100

                    print("epoch-iteration:{}-{}, loss:{}, accuracy:{}".format(epoch + 1, (batch_cnt + 1), loss_sum100,
                                                                               acc_sum100))
                    print("time:{}".format(datetime.datetime.now() - start_time))

                    LossLine = [np.append(LossLine[0], loss_sum100[0]), np.append(LossLine[0], loss_sum100[0])]
                    AccLine = [np.append(AccLine[0], acc_sum100[0]), np.append(AccLine[1], acc_sum100[1])]
                    if save_output:
                        np.save(save_dir + '/trainLossLine', LossLine)
                        np.save(save_dir + '/trainAccLine', AccLine)
                    acc_sum100 = [0, 0]
                    loss_sum100 = [0, 0]

            print("第{}个epoch训练完成！".format(epoch + 1))
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


# inception 优化规则https://blog.csdn.net/docrazy5351/article/details/78993306


def main(is_test=False, pth_file='',random_state=1,fold_index=2):
    note = 'InceptionV3 is_test={}'.format(is_test)  # 直接跑
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
    # 定义一个哈希表

    GPU_device = True
    if not torch.cuda.is_available():
        GPU_device = False

    batchsize = 32
    numworks = 0
    model = Inception3()

    if not pth_file == '':
        model.load_state_dict(torch.load(pth_file, map_location='cpu'), strict=True)

    if GPU_device == True:
        model.cuda()

    print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(model),
                                                             count_parameters(model) / 11689512))
    print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(model, grad=False),
                                                             count_parameters(model, grad=False) / 11689512))

    print('')

    macs, params = get_model_complexity_info(model, (1, 100, 250), as_strings=False,
                                             print_per_layer_stat=False, verbose=True, ost=log1)
    print('{:<30}  {:<8}'.format('MACs: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    base_lr = 0.0008
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.00001)
    # criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss()

    if is_test:
        dataset_dir0324 = r'E:\dataset\原始时空矩阵降采样增广0324qiaoji分训练测试/test/'
        dataset_dirwajue = r'E:\dataset\原始时空矩阵降采样增广wajue分训练测试/test/'
    else:
        dataset_dir0324 = r'E:\dataset\原始时空矩阵降采样增广0324qiaoji分训练测试/train/'
        dataset_dirwajue = r'E:\dataset\原始时空矩阵降采样增广wajue分训练测试/train/'

    dataset_MTL = Dataset_mat_MTL(dataset_dir0324=dataset_dir0324, dataset_dirwajue=dataset_dirwajue, ram=True,
                                  paper_single=True,is_test=is_test,random_state=random_state,fold_index=fold_index)

    dataloader1 = {}
    dataloader1['train'] = torch.utils.data.DataLoader(dataset_MTL.dataset['train'], batch_size=batchsize,
                                                       shuffle=True, num_workers=numworks)
    dataloader1['val'] = torch.utils.data.DataLoader(dataset_MTL.dataset['val'], batch_size=batchsize,
                                                     shuffle=True, num_workers=numworks)

    # 保存样本名-标签列表
    dataset_MTL.dataset['train'].get_name_label_csv(savedir='train name label.csv', paper_single=True)
    dataset_MTL.dataset['val'].get_name_label_csv(savedir='val name label.csv', paper_single=True)

    is_train = True
    if is_train:
        train(model=model, data_loader=dataloader1, epoch_num=41, start_epoch=0, optimizer=optimizer,
              criterion=criterion, use_gpu=GPU_device, save_dir=save_dir, save_output=True,is_test=is_test)

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
        leibie_event = ['Striking', 'Digging']
        leibei_distance = ['{}m'.format(i) for i in range(16)]
        cm_list = glob.glob(save_dir + 'confusion matrix*.npy')
        for cm in cm_list:
            mat1 = np.load(cm)
            if len(mat1[0]) == 16:
                draw_confusion_matrix(confusion_matrix=mat1, leibie1=leibei_distance, figsize=(7, 6.8),
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
            f.write('\n'+note+'\n')
            f.write(str(np.max(line, axis=1)[0]))
            f.write('\n')
            f.write(str(np.max(line, axis=1)[1]))


if __name__ == '__main__':
    main()
