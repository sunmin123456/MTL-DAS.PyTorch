import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold
from torch import optim
from torch.autograd import Variable
import torch
import os
# 训练/验证过程
import torch.nn as nn
from tqdm import tqdm
import scipy.io as io


class DataCollector:
    """
    A class for dataset deployment and easy access
    Usage:
        dataCollector_striking = DataCollector(dataset_dir, ['data'])   # Initialization
        dataCollector_striking.get_all_categorys()  # Get the category list
        dataCollector_striking.get_fileFullnameList_by_category('0m')  # Get all the file paths for one category
        dataCollector_striking.get_mat_by_categoryIndex('0m', 1)  # Get the sample data by the category and index
    """

    def __init__(self, dirPath, keyList):
        """
        Initialization
        :param dirPath: Dataset path
        :param keyList: The possible keys in the mat files. In our dataset, only ‘data’ is used.
        """

        self.dirPath = dirPath
        self.keyList = keyList

        self.allFileFullNameDict = {}

        for category in self.get_all_categorys():
            self.allFileFullNameDict[category] = self.get_fileFullnameList_by_category(category)

    def get_all_categorys(self):
        import re
        category_dir_list = os.listdir(self.dirPath)
        category_dir_list = sorted(category_dir_list, key=lambda i: int(re.findall(r'\d+', i)[0]))
        return category_dir_list

    def get_fileFullnameList_by_category(self, categoryName):
        fileNameList = os.listdir(self.dirPath + '/' + categoryName)
        filePath = self.dirPath + '/' + categoryName + '/'
        fileFullNameList = [filePath + name for name in fileNameList]
        return fileFullNameList

    def get_one_mat(self, fileFullName):
        import scipy.io as io
        data1 = io.loadmat(fileFullName)
        return_data = None

        if len(self.keyList) == 1:
            return_data = data1[self.keyList[0]]
        else:
            for k, v in data1.items():
                for key in self.keyList:
                    if k == key:
                        return_data = v

        if return_data is not None:
            return return_data
        else:
            raise ValueError

    def get_mat_by_categoryIndex(self, category, index):
        fileFullName = self.allFileFullNameDict[category][index]
        # print(fileFullName)
        return self.get_one_mat(fileFullName)

    def get_mat_name_by_categoryIndex(self, category, index):
        fileFullName = self.allFileFullNameDict[category][index]

        return self.get_one_mat(fileFullName), fileFullName


def add_gaussian(signal, SNR=8):
    np.random.seed(1)
    SNR = SNR
    noise = np.random.randn(len(signal))  # 产生N(0,1)噪声数据
    noise = noise - np.mean(noise)  # 均值为0
    signal_power = np.linalg.norm(signal - signal.mean()) ** 2 / len(signal)  # 此处是信号的std**2
    noise_variance = signal_power / np.power(10, (SNR / 10))  # 此处是噪声的std**2
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise  ##此处是噪声的std**2
    signal_noise = noise + signal

    Ps = (np.linalg.norm(signal - signal.mean())) ** 2  # signal power
    Pn = (np.linalg.norm(signal - signal_noise)) ** 2  # noise power
    snr = 10 * np.log10(Ps / Pn)

    # print(Ps)
    # print(Pn)
    # print(snr)
    return signal_noise

    # ————————————————
    # 版权声明：本文为CSDN博主「迪普达克范特西」的原创文章，遵循CC
    # 4.0
    # BY - SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https://blog.csdn.net/sinat_24259567/article/details/93889547


# 给信号加入指定功率的噪声
def add_gaussian_refer(signal, SNR=10, Pn_origin=0.02, Pn=0.2):
    np.random.seed(1)
    SNR = SNR
    noise = np.random.randn(len(signal))  # 产生N(0,1)噪声数据
    noise = noise - np.mean(noise)  # 均值为0
    signal_power = np.linalg.norm(signal - signal.mean()) ** 2 / len(signal)  # 此处是信号的std**2
    # noise_variance = signal_power / np.power(10, (SNR / 10))  # 此处是噪声的std**2 或噪声功率
    noise_variance = Pn

    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise  ##此处是噪声的std**2
    signal_noise = noise + signal

    Ps = (np.linalg.norm(signal - signal.mean())) ** 2 / len(signal)  # signal power
    Pn = (np.linalg.norm(signal - signal_noise)) ** 2 / len(signal)  # noise power
    snr = 10 * np.log10(Ps / Pn)

    # print(Ps)
    # print(Pn)
    # print(snr)
    return signal_noise


class Dataset_mat_MTL():  # 修改这个函数
    '''
    example:
    get_fftmap_dataset(dataset_dir=r'C:/dataset\0324qiaoji/')
    dataset1 = Dataset(fftmapDatasetPath=r'D:\experiment_preparation\DL_recognition\fftmapDataset\fftmapDataset.npy',
                       fftmapDatasetLabelPath=r'D:\experiment_preparation\DL_recognition\fftmapDataset\fftmapDatasetLabel.npy')
    for data, label in dataset1.dataset['train']:
        print(label)
    '''

    # 把is_test改成True，使用全训练集
    # 0.17647
    def __init__(self, dataset_dir_striking, dataset_dir_excavating, testRate=0.17647, random_state=1,
                 category_dir_list0324=None,
                 category_dir_listwajue=None, ram=False, multi_categories=False, is_test=False, fold_index=None):
        from sklearn.model_selection import train_test_split

        dataCollector_striking = DataCollector(dataset_dir_striking, ['data'])
        dataCollector_excavating = DataCollector(dataset_dir_excavating, ['data'])

        if category_dir_list0324 is None:
            category_dir_list0324 = dataCollector_striking.get_all_categorys()
        if category_dir_listwajue is None:
            category_dir_listwajue = dataCollector_excavating.get_all_categorys()

        # 生成样本-标签对
        self.matpathListTrain = []
        self.labelListTrain = []
        self.matpathListTest = []
        self.labelListTest = []

        for category1 in category_dir_list0324:
            filenum = dataCollector_striking.get_fileFullnameList_by_category(category1)

            if is_test:  # 测试

                for filepath in filenum:
                    self.matpathListTrain.append(filepath)
                    self.labelListTrain.append([int(category1[:-1]), 0])  # 0表示0324，0表示0m

                for filepath in filenum:
                    self.matpathListTest.append(filepath)
                    self.labelListTest.append([int(category1[:-1]), 0])  # 0表示0324，0表示0m


            else:  # 训练

                if fold_index is None:
                    train_filepath_one_category, test_filepath_one_category = train_test_split(filenum,
                                                                                               test_size=testRate,
                                                                                               random_state=random_state)

                else:  # 采用5折交叉的方法

                    KF = KFold(n_splits=5, shuffle=True, random_state=1)
                    train_filepath_one_category_list = []
                    test_filepath_one_category_list = []
                    for train_index, test_index in KF.split(filenum):
                        train_filepath_one_category_list.append(train_index)
                        test_filepath_one_category_list.append(test_index)
                    train_filepath_one_category = [filenum[aa] for aa in train_filepath_one_category_list[fold_index]]
                    test_filepath_one_category = [filenum[aa] for aa in test_filepath_one_category_list[fold_index]]

                for filepath in train_filepath_one_category:
                    self.matpathListTrain.append(filepath)
                    self.labelListTrain.append([int(category1[:-1]), 0])  # 0表示0324，0表示0m

                for filepath in test_filepath_one_category:
                    self.matpathListTest.append(filepath)
                    self.labelListTest.append([int(category1[:-1]), 0])  # 0表示0324，0表示0m

        for category1 in category_dir_listwajue:
            filenum = dataCollector_excavating.get_fileFullnameList_by_category(category1)

            if is_test:  # 测试

                for filepath in filenum:
                    self.matpathListTrain.append(filepath)
                    self.labelListTrain.append([int(category1[:-1]), 1])  # 0表示0324，0表示0m

                for filepath in filenum:
                    self.matpathListTest.append(filepath)
                    self.labelListTest.append([int(category1[:-1]), 1])  # 0表示0324，0表示0m

            else:  # 训练

                if fold_index is None:
                    train_filepath_one_category, test_filepath_one_category = train_test_split(filenum,
                                                                                               test_size=testRate,
                                                                                               random_state=random_state)
                else:
                    KF = KFold(n_splits=5, shuffle=True, random_state=1)
                    train_filepath_one_category_list = []
                    test_filepath_one_category_list = []
                    for train_index, test_index in KF.split(filenum):
                        train_filepath_one_category_list.append(train_index)
                        test_filepath_one_category_list.append(test_index)

                    train_filepath_one_category = [filenum[aa] for aa in train_filepath_one_category_list[fold_index]]
                    test_filepath_one_category = [filenum[aa] for aa in test_filepath_one_category_list[fold_index]]

                for filepath in train_filepath_one_category:
                    self.matpathListTrain.append(filepath)
                    self.labelListTrain.append([int(category1[:-1]), 1])  # 0表示0324，0表示0m

                for filepath in test_filepath_one_category:
                    self.matpathListTest.append(filepath)
                    self.labelListTest.append([int(category1[:-1]), 1])  # 0表示0324，0表示0m

        self.dataset = {}

        # Transform the categories of two tasks to multi-categories
        if multi_categories:
            label_length = len(self.labelListTrain)
            for i in range(label_length):
                self.labelListTrain[i] = self.labelListTrain[i][0] + 16 * self.labelListTrain[i][1]

            label_length = len(self.labelListTest)
            for i in range(label_length):
                self.labelListTest[i] = self.labelListTest[i][0] + 16 * self.labelListTest[i][1]

            if ram:
                self.dataset['train'] = Datasetram(self.matpathListTrain, self.labelListTrain, paper_single=True)
                self.dataset['val'] = Datasetram(self.matpathListTest, self.labelListTest, paper_single=True)
            else:
                self.dataset['train'] = DatasetDisk(self.matpathListTrain, self.labelListTrain, paper_single=True)
                self.dataset['val'] = DatasetDisk(self.matpathListTest, self.labelListTest, paper_single=True)
            return

        if ram:
            self.dataset['train'] = Datasetram(self.matpathListTrain, self.labelListTrain)
            self.dataset['val'] = Datasetram(self.matpathListTest, self.labelListTest)
        else:
            self.dataset['train'] = DatasetDisk(self.matpathListTrain, self.labelListTrain)
            self.dataset['val'] = DatasetDisk(self.matpathListTest, self.labelListTest)


# 数据处理程序
def data_process(mat):
    # 降采样
    # mat = mat[:, :, np.arange(0, 63, 2)]
    # mat = mat[np.arange(0, 40, 2), :,:]
    # mat = mat[:, np.arange(0, 50, 2),:]

    # 降采样，用于原始时空矩阵
    mat = mat[:, np.arange(0, 500, 2)]

    # 加噪
    for i in range(100):
        mat[i] = add_gaussian_refer(mat[i], Pn=0.045)  # -10dB  原本0.015

    # 0-1归一化
    # mat = (mat - np.min(mat)) / (
    #         np.max(mat) - np.min(mat))

    # # 取对数
    # mat = np.log(mat + 0.001)
    # # 0-1归一化
    # mat = (mat - np.min(mat)) / (
    #         np.max(mat) - np.min(mat))

    mat = mat[np.newaxis, :]
    mat = mat.astype(np.float32)
    return mat


class Datasetram(torch.utils.data.Dataset):  # 修改这个函数
    def __init__(self, mat_list, label_list, key='data', paper_single=False):
        assert len(mat_list) == len(label_list)

        self.mat_list = mat_list
        self.label_list = label_list
        self.key = key
        self.mat_file_list = []  # 一个很大的矩阵，用来存储mat文件
        self.paper_single = paper_single

        # 读取数据
        for i in tqdm(range(len(self.mat_list))):
            # 这里只将mat文件放到内存
            mat = io.loadmat(self.mat_list[i])[self.key]
            mat = data_process(mat)
            self.mat_file_list.append(mat)

    def __len__(self):
        return len(self.mat_list)

    def __getitem__(self, item):
        if self.paper_single:
            return self.mat_file_list[item], self.label_list[item]
        return self.mat_file_list[item], self.label_list[item][0], self.label_list[item][1]

    # 12.26新增方法
    def get_name_label_csv(self, savedir='./name_label.csv', paper_single=False):
        import pandas as pd

        distance_list = []
        event_list = []

        if paper_single:
            csv_dict = {
                'mat name': self.mat_list,
                'label': self.label_list
            }

        else:
            for label in self.label_list:
                distance_list.append(label[0])
                event_list.append(label[1])
            csv_dict = {
                'mat name': self.mat_list,
                'distance label': distance_list,
                'event label': event_list
            }
        csv_table = pd.DataFrame(csv_dict)
        csv_table.to_csv(savedir, encoding='gbk')


class DatasetDisk(torch.utils.data.Dataset):  # 修改这个函数
    def __init__(self, mat_list, label_list, key='data', paper_single=False):
        assert len(mat_list) == len(label_list)
        self.mat_list = mat_list
        self.label_list = label_list
        self.key = key
        self.paper_single = paper_single

    def __len__(self):
        return len(self.mat_list)

    def __getitem__(self, item):
        mat = io.loadmat(self.mat_list[item])[self.key]
        mat = data_process(mat)

        if self.paper_single:
            return mat, self.label_list[item]

        [distance_label, event_label] = self.label_list[item]

        return mat, distance_label, event_label

    # 12.26新增方法
    def get_name_label_csv(self, savedir='./name_label.csv', paper_single=False):
        import pandas as pd

        distance_list = []
        event_list = []

        if paper_single:
            csv_dict = {
                'mat name': self.mat_list,
                'label': self.label_list
            }

        else:
            for label in self.label_list:
                distance_list.append(label[0])
                event_list.append(label[1])
            csv_dict = {
                'mat name': self.mat_list,
                'distance label': distance_list,
                'event label': event_list
            }
        csv_table = pd.DataFrame(csv_dict)
        csv_table.to_csv(savedir, encoding='gbk')


if __name__ == '__main__':
    dataset_dir0324 = r"\\121.48.161.226\kb208datapool\LabFiles\users\wyf\数据集-偶尔用\空间时频图数据集（0324敲击）/"
    dataset_dirwajue = r"\\121.48.161.226\kb208datapool\LabFiles\users\wyf\数据集-偶尔用\空间时频图（归一化同态滤波挖掘）/"

    dataset_MTL = Dataset_mat_MTL(dataset_dir_striking=dataset_dir0324, dataset_dir_excavating=dataset_dirwajue)

    for data in dataset_MTL.dataset['train']:
        pass
